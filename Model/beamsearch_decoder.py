from RNN import EncoderRNN, DecoderRNN, Q_predictor
import sys
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import os
import logging
import wandb
import operator

import wandb
sys.path.append('../Utils/')

parser = argparse.ArgumentParser()
#parser.add_argument('--task')
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--dataset', type=str, default="frames")
parser.add_argument('--loss',type=str, default='combine') #nll, bert, combine, alternate
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--type', type=str, default='train') #train, valid, test
parser.add_argument('--alpha',type=float, default=0.0)
parser.add_argument('--toggle_loss',type=float, default=0.5)
parser.add_argument('--teacher_forcing', type=float, default=0.1)
parser.add_argument('--change_nll_mask', action='store_true')
parser.add_argument('--save_base', type=str, default='..')
parser.add_argument('--encoder_learning_rate', type=float, default=0.004)
parser.add_argument('--decoder_learning_rate', type=float, default=0.004)
parser.add_argument('--output_dropout', type=float,default=0.0)
parser.add_argument('--data_path', type=str, default="../Dataset")
parser.add_argument('--save_every_epoch', action='store_true')
parser.add_argument('--reload', action='store_true')
parser.add_argument('--start_epoch', type=int, default=-1)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--beam_search', action='store_true')
parser.add_argument('--topk', type=int, default=5)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--beam_width',type = int,default = 5)
parser.add_argument('--no_posteos_mask', action='store_true') #if true, don't mask the words generated after the <eos> token
#if true, don't apply the mask before generating the Bert sentence (allow the model to generate masked tokens, and then mask them during the embedding calculation)
parser.add_argument('--no_prebert_mask', action='store_true')
parser.add_argument('--wandb_project', type=str, default='metadial')
parser.add_argument('--validation_model', type=str, default='start_epoch') #which model to use for validation/test, 'best_mle' or 'best_combined' or the model of epoch 'start_epoch'
args = parser.parse_args()

save_path = os.path.join(args.save_base, 'MetaDial')
result_path = os.path.join(save_path, 'Results', args.dataset)

sys.path.append(args.data_path)
from WoZ_data_iterator import WoZGraphDataset
from Frames_data_iterator import FramesGraphDataset
from Bert_util import Load_embeddings, Bert_loss, Mask_sentence, getTopK, BeamSearchNode
from nltk.translate.meteor_score import meteor_score
from queue import PriorityQueue

if not os.path.exists(os.path.join(result_path, 'Samples')):
    os.makedirs(os.path.join(result_path, 'Samples'))

saved_models = os.path.join(save_path, 'Saved_Models')
if not os.path.exists(saved_models):
    os.makedirs(saved_models)

log_path = os.path.join(save_path, 'logs')
if not os.path.exists(log_path):
    os.makedirs(log_path)

# tensorboard_path = os.path.join(save_path, 'tensorboard')
# if not os.path.exists(tensorboard_path):
#     os.makedirs(tensorboard_path)

fname = os.path.join(result_path, 'Log.txt')
samples_fname = os.path.join(result_path, 'Samples')

try:
    saver = open(fname,'a')
except:
    saver = open(fname,'w')
    if args.type == 'valid':
        sample_saver = open(samples_fname,'w')
    saver.close()

# Hyper-parameters
config = {}
config['data_path'] = args.data_path
config['save_path'] = save_path
# config['tensorboard_path'] = tensorboard_path
config["wandb_project"] = args.wandb_project
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config['seed'] = args.seed

if args.dataset == 'mwoz':
    config['data'] = WoZGraphDataset(Data_dir=config['data_path'] + '/MULTIWOZ2/')
else:
    config['data'] = FramesGraphDataset(Data_dir=config['data_path'] + '/Frames-dataset/')

# Hyper-parameters
config['teacher_forcing'] = args.teacher_forcing
config['sequence_length'] = 101
config['input_size'] = config['data'].vlen
config['hidden_size'] = args.hidden_size
config['num_layers'] = 1
config['output_size'] = config['data'].vlen
config['num_epochs'] = args.num_epochs
config['output_dropout'] = args.output_dropout
config['change_nll_mask'] = args.change_nll_mask
config['decoder_learning_rate'] = args.decoder_learning_rate
config['encoder_learning_rate'] = args.encoder_learning_rate
config['batch_size'] = args.batch_size
config['alpha'] = args.alpha
config['toggle'] = args.toggle_loss
config['loss'] = args.loss
config['dataset'] = args.dataset
config['device'] = device
config['save_every_epoch'] = args.save_every_epoch
config['posteos_mask'] = ~args.no_posteos_mask
config['prebert_mask'] = ~args.no_prebert_mask
config['best_mle_valid'] = 10000
config['best_combined_loss'] = 10000
config['meteor_valid'] = 0
if config['prebert_mask']:
    config['id'] = '{}_preBertMask_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.dataset,args.hidden_size,args.encoder_learning_rate,
                                                                         args.decoder_learning_rate,args.loss,args.alpha,
                                                                         args.toggle_loss,args.output_dropout,args.change_nll_mask, args.no_posteos_mask)
else:
    config['id'] = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.dataset,args.hidden_size,args.encoder_learning_rate,
                                                             args.decoder_learning_rate,args.loss,args.alpha,args.toggle_loss,
                                                             args.output_dropout,args.change_nll_mask, args.no_posteos_mask)
config['wandb_id'] = config['id'] + '_' + str(np.random.randint(1000))

config['weights'] = np.hstack([np.array([1,1,1,0]),np.ones(config['input_size']-4)])
config['pad_index'] = 3
config['eos_index'] = 1
config['epoch'] = 0
torch.manual_seed(config['seed'])
np.random.seed(config['seed'])
# 1 => account for loss
# 0 => mask the token
# embedding matrix Nxd


class Seq2Seq(nn.Module):
    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        #Encoder_model can be s2s or hred
        self.Data = config['data']
        self.config = config
        self.Encoder = EncoderRNN(self.config['input_size'], self.config['hidden_size'], self.config['num_layers']).to(self.config['device'])
        self.Decoder = DecoderRNN(self.config['hidden_size'], self.config['output_size'], self.config['input_size'], self.config['num_layers']).to(self.config['device'])
        _, self.weights = Load_embeddings(config['dataset'], )
        self.Bert_embedding = nn.Embedding.from_pretrained(self.weights, freeze=True).to(self.config['device'])
        # Loss and optimizer
        self.criterion = nn.NLLLoss(weight=torch.from_numpy(config['weights']).float()).to(self.config['device'])
        # criterion_2 = nn.CrossEntropyLoss().to(self.config['device'])

        self.optimizer = torch.optim.RMSprop(self.Encoder.parameters(), lr=config['encoder_learning_rate'],alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        self.optimizer_dec = torch.optim.Adam(self.Decoder.parameters(), lr =config['decoder_learning_rate'])
        self.Opts = [self.optimizer, self.optimizer_dec]

        # self.writer = SummaryWriter(os.path.join(config['tensorboard_path'], config['id']))

    def modelrun(self, Data='', type_='train', total_step=200, ep=0, sample_saver=''):
        loss_mle_inf = 0.
        loss_bert_inf = 0.
        train_loss_inf = 0.

        self.Data = Data
        self.sample_saver = sample_saver
        meteor_score_valid = 0.
        count_examples = 0.
        for i in range(total_step):
            seq_loss_a = 0.
            batch_size = self.Data[i]['input'].shape[0]
            count_examples += batch_size
            hidden_enc = (torch.zeros(self.config['num_layers'], batch_size, self.config['hidden_size'], device=self.config['device']), torch.zeros(self.config['num_layers'], batch_size, self.config['hidden_size'], device=self.config['device']))

            input_ = torch.from_numpy(self.Data[i]['input']).to(self.config['device']).view(batch_size,self.config['sequence_length'],self.config['input_size'])
            decoder_input = torch.from_numpy(self.Data[i]['target']).to(self.config['device']).view(batch_size,self.config['sequence_length'],self.config['input_size'])

            # if type_ == 'valid':
            response_ = []
            context_ = []
            target_response = []
            response_premasked = [] #the response generated by choosing only unmasked words
            # Generate random masks
            print(i,'/',total_step)
            topk = args.topk
            decoded_batch = []
            for di in range(self.Data[i]['encoder_length']):
                hidden_enc, out = self.Encoder(input_[:,di,:].view(-1,1,self.config['input_size']),hidden_enc)
                context_ = context_ + [torch.argmax(input_[:,di,:].view(batch_size,-1),dim =1).view(-1,1)]
                target_response = target_response + [torch.argmax(decoder_input[:,di,:].view(batch_size,-1),dim =1).view(-1,1)]
            for b_ind in range(self.Data[i]['input'].shape[0]):
                decoder_hidden = (hidden_enc[0][:,b_ind, :].unsqueeze(0),hidden_enc[1][:,b_ind, :].unsqueeze(0))
                decoder_input_ = decoder_input[b_ind,0,:]

                node = BeamSearchNode(decoder_hidden, None, torch.argmax(decoder_input_).item(), 0, 1,decoder_input_.view(-1,1,self.config['input_size']))
                nodes = PriorityQueue()
                endnodes = []
                number_required = min((topk + 1), topk - len(endnodes))
                nodes.put((-node.eval(), node))
                qsize = 1
                while True:
                    # give up when decoding takes too long
                    if qsize > 2000: break

                    score, n = nodes.get()
                    decoder_input_ = n.decoder_input_d
                    decoder_hidden = n.h
                    if n.wordid == 1 and n.prevNode != None:
                        endnodes.append((score, n))
                        # if we reached maximum # of sentences required
                        if len(endnodes) >= number_required:
                            break
                        else:
                            continue

                    decoder_output, decoder_hidden = self.Decoder(decoder_input_, decoder_hidden)
                    log_prob, indexes = torch.topk(decoder_output.view(-1), args.beam_width)
                    nextnodes = []
                    print(indexes)
                    for new_k in range(args.beam_width):
                        decoded_t = indexes[new_k].item()
                        print(decoded_t)
                        log_p = log_prob[new_k].item()
                        node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1, decoder_output.view(-1,1,self.config['input_size']))
                        score = -node.eval()
                        nextnodes.append((score, node))
                    for m in range(len(nextnodes)):
                        score, nn = nextnodes[m]
                        nodes.put((score, nn))
                        # increase qsize
                    qsize += len(nextnodes) - 1
                # choose nbest paths, back trace them
                if len(endnodes) == 0:
                    endnodes = [nodes.get() for _ in range(topk)]
                utterances = []
                for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                    utterance = []
                    utterance.append(n.wordid)
                    # back trace
                    while n.prevNode != None:
                        n = n.prevNode
                        utterance.append(n.wordid)

                    utterance = utterance[::-1]
                    utterances.append(utterance)

                decoded_batch.append(utterances)
            con = torch.cat(context_, dim=1)
            tar = torch.cat(target_response, dim=1)

            if type_ !='train':
                for c_index in range(con.shape[0]):
                    c = ' '.join([self.Data.Vocab_inv[idx.item()] for idx in con[c_index]])
                    t_list = [self.Data.Vocab_inv[idx.item()] for idx in tar[c_index]]
                    t = ' '.join(t_list)
                    r = ''
                    for beam_ind in range(len(decoded_batch[c_index])):
                        res = decoded_batch[c_index][beam_ind]
                        res_str = ' '.join([self.Data.Vocab_inv[idx] for idx in res])
                        r += 'Model_Response_'+str(beam_ind)+': '+res_str+'\n'
                    self.sample_saver.write('Context: '+ c + '\n' + r + 'Target: ' + t + '\n\n')
            if type_ == 'train':
                self.optimizer.zero_grad()
                self.optimizer_dec.zero_grad()

                train_loss.backward()
                for O_ in self.Opts:
                    O_.step()

        if type_ == 'eval':
            logging.info(
                f"Train:   Loss_MLE_eval: {loss_mle_inf:.4f},  Loss_Bert_eval: {loss_bert_inf:.4f}, Meteor_score_valid : {meteor_score_valid:.4f}\n")
            wandb.log({'Loss_MLE_eval': loss_mle_inf, 'Loss_Bert_eval': loss_bert_inf, 'Meteor_score_valid': meteor_score_valid,'train_loss_eval': train_loss_inf, 'global_step':ep})
            return loss_mle_inf, train_loss_inf, meteor_score_valid/count_examples
        if type_ == 'train':
            logging.info(
                f"Train:   Loss_MLE_train: {loss_mle_inf:.4f},  Loss_MLE_train: {loss_bert_inf:.4f}\n")
            wandb.log({'Loss_MLE_train': loss_mle_inf, 'Loss_Bert_train': loss_bert_inf, 'train_loss_train': train_loss_inf, 'global_step':ep})


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if args.dataset == 'mwoz' and (args.type == 'train' or args.type == 'valid'):
        Data_train = WoZGraphDataset()
        Data_valid = WoZGraphDataset(suffix = 'valid')
    elif args.dataset == 'mwoz' and args.type == 'test':
        Data_test = WoZGraphDataset(suffix='test')
    elif args.dataset == 'frames' and (args.type == 'train' or args.type == 'valid'):
        Data_train = FramesGraphDataset()
        Data_valid = FramesGraphDataset(suffix='valid')
    elif args.dataset == 'frames' and args.type == 'test':
        Data_test = FramesGraphDataset(suffix='test')

    Model = Seq2Seq(config)

    if args.type == 'test':
        if args.validation_model == 'best_combined':
            sample_saver_test = open(samples_fname + "_test_" + str(args.topk) + '_' + str(args.beam_width) + '_' + config['id'] + '_best_combined.txt', 'w')
            sample_saver_test = open(samples_fname + "_test_" + str(args.topk) + '_' + str(args.beam_width) + '_' + config['id'] + '_best_combined.txt', 'a')
            checkpoint = torch.load(os.path.join(saved_models, config['id'] + '_best_combined_loss'))
        elif args.validation_model == 'best_mle':
            sample_saver_test = open(samples_fname + "_test_" + str(args.topk) + '_' + str(args.beam_width) + '_' + config['id'] + '_best_mle.txt', 'w')
            sample_saver_test = open(samples_fname + "_test_" + str(args.topk) + '_' + str(args.beam_width) + '_' + config['id'] + '_best_mle.txt', 'a')
            checkpoint = torch.load(os.path.join(saved_models, config['id'] + '_best_mle_valid'))
        elif str(args.validation_model) == 'best_meteor':
            sample_saver_test = open(samples_fname + "_test_" + str(args.topk) + '_' + str(args.beam_width) + '_' + config['id'] + '_best_mle.txt', 'w')
            sample_saver_test = open(samples_fname + "_test_" + str(args.topk) + '_' + str(args.beam_width) + '_' + config['id'] + '_best_mle.txt', 'a')
            checkpoint = torch.load(os.path.join(saved_models, config['id'] + '_meteor_valid'))
        else:
            sample_saver_test = open(samples_fname + "_test_" + str(args.topk) + '_' + str(args.beam_width) + '_' + config['id'] + '_' + str(args.start_epoch) + '.txt', 'w')
            sample_saver_test = open(samples_fname + "_test_" + str(args.topk) + '_' + str(args.beam_width) + '_' + config['id'] + '_' + str(args.start_epoch) + '.txt', 'a')
            checkpoint = torch.load(os.path.join(saved_models, config['id'] + '_' + str(args.start_epoch)))
        Model.load_state_dict(checkpoint['model_State_dict'])
        Model.modelrun(Data=Data_test, type_='valid', total_step=Data_test.num_batches, ep=0,sample_saver=sample_saver_test)
