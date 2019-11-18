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
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--no_posteos_mask', action='store_true') #if true, don't mask the words generated after the <eos> token
#if true, don't apply the mask before generating the Bert sentence (allow the model to generate masked tokens, and then mask them during the embedding calculation)
parser.add_argument('--no_prebert_mask', action='store_true')
parser.add_argument('--wandb_project', type=str, default='metadial')
parser.add_argument('--dont_use_best', action='store_true')

args = parser.parse_args()

save_path = os.path.join(args.save_base, 'MetaDial')
result_path = os.path.join(save_path, 'Results', args.dataset)

sys.path.append(args.data_path)
from WoZ_data_iterator import WoZGraphDataset
from Frames_data_iterator import FramesGraphDataset
from Bert_util import Load_embeddings, Bert_loss, Mask_sentence


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
torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

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
config['loss'] = args.loss
config['dataset'] = args.dataset
config['device'] = device
config['save_every_epoch'] = args.save_every_epoch
config['posteos_mask'] = ~args.no_posteos_mask
config['prebert_mask'] = ~args.no_prebert_mask
config['best_mle_valid'] = 0.0
if config['prebert_mask']:
    config['id'] = '{}_preBertMask_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.dataset,args.hidden_size,args.encoder_learning_rate,args.decoder_learning_rate,args.loss,args.alpha,args.toggle_loss,args.output_dropout,args.change_nll_mask, args.no_posteos_mask)
else:
    config['id'] = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.dataset,args.hidden_size,args.encoder_learning_rate,args.decoder_learning_rate,args.loss,args.alpha,args.toggle_loss,args.output_dropout,args.change_nll_mask, args.no_posteos_mask)

config['weights'] = np.hstack([np.array([1,1,1,0]),np.ones(config['input_size']-4)])
config['pad_index'] = 3
config['eos_index'] = 1
config['epoch'] = 0
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

        self.Data = Data
        self.sample_saver = sample_saver
        for i in range(total_step):
            seq_loss_a = 0.
            batch_size = self.Data[i]['input'].shape[0]
            hidden_enc = (torch.zeros(self.config['num_layers'], batch_size, self.config['hidden_size'], device=self.config['device']), torch.zeros(self.config['num_layers'], batch_size, self.config['hidden_size'], device=self.config['device']))

            input_ = torch.from_numpy(self.Data[i]['input']).to(self.config['device']).view(batch_size,self.config['sequence_length'],self.config['input_size'])
            decoder_input = torch.from_numpy(self.Data[i]['target']).to(self.config['device']).view(batch_size,self.config['sequence_length'],self.config['input_size'])

            # if type_ == 'valid':
            response_ = []
            context_ = []
            target_response = []
            response_premasked = [] #the response generated by choosing only unmasked words
            # Generate random masks

            for di in range(self.Data[i]['encoder_length']):
                hidden_enc, out = self.Encoder(input_[:,di,:].view(-1,1,self.config['input_size']),hidden_enc)
                context_ = context_ + [torch.argmax(input_[:,di,:].view(batch_size,-1),dim =1).view(-1,1)]
            decoder_hidden = hidden_enc

            decoder_input_ = decoder_input[:,0,:]
            dec_list = []
            mask = torch.from_numpy(config['weights']).to(device).bool()

            if args.type == 'train' and config['output_dropout'] > 0:
                weight_random = np.random.random(len(config['weights'])-4) > config['output_dropout']
                config['weights'] = np.hstack([config['weights'][:4],weight_random.astype(int)])
                mask = torch.from_numpy(config['weights']).to(device).bool()
                if config['change_nll_mask']:
                    self.criterion = nn.NLLLoss(weight = torch.from_numpy(config['weights']).float()).to(device)

            for di in range(self.Data[i]['decoder_length']-1):
                decoder_output, decoder_hidden = self.Decoder(decoder_input_.view(-1,1,self.config['input_size']), decoder_hidden)
                if np.random.rand() > self.config['teacher_forcing'] and type_ == 'train':
                    decoder_input_ = decoder_output.view(-1,1,self.config['input_size'])
                else:
                    decoder_input_ = decoder_input[:,di+1,:].view(-1,1,self.config['input_size'])
                seq_loss_a += self.criterion(input=decoder_output[:,-1,:], target=torch.max(decoder_input[:,di+1,:], dim=1)[-1])
                dec_list+=[decoder_output.view(-1,1,self.config['input_size'])]
                target_response = target_response + [torch.argmax(decoder_input[:,di,:].view(batch_size,-1),dim =1).view(-1,1)]
                response_ = response_ + [torch.argmax(decoder_output.view(batch_size,-1),dim =1).view(-1,1)]

                if args.type == 'train':
                    response_premasked = response_premasked + [torch.argmax(decoder_output.view(batch_size,-1).masked_fill(~mask, -10**6),dim =1).view(-1,1)]
                else:
                    response_premasked = response_
            con = torch.cat(context_, dim=1)
            res = torch.cat(response_, dim=1)
            res_premasked = torch.cat(response_premasked, dim=1)
            tar = torch.cat(target_response, dim=1)

            loss = seq_loss_a/batch_size/self.Data[i]['decoder_length']
            loss_mle_inf += loss.item()/total_step#seq_loss_a.item()/batch_size/self.Data[i]['decoder_length']

            if config['prebert_mask']:
                loss_bert = Bert_loss(self.Bert_embedding(res_premasked), self.Bert_embedding(tar))
            else:
                res_postmasked = Mask_sentence(res, mask, config)
                loss_bert = Bert_loss(self.Bert_embedding(res_postmasked), self.Bert_embedding(tar))
            loss_bert_inf += loss_bert.item()/total_step

            if type_ == 'valid' or type_ == 'test':
                for c_index in range(con.shape[0]):
                    c = ' '.join([self.Data.Vocab_inv[idx.item()] for idx in con[c_index]])
                    r = ' '.join([self.Data.Vocab_inv[idx.item()] for idx in res[c_index]])
                    t = ' '.join([self.Data.Vocab_inv[idx.item()] for idx in tar[c_index]])
                    self.sample_saver.write('Context: '+ c + '\n' + 'Model_Response: ' + r + '\n' + 'Target: ' + t + '\n\n')
            if type_ == 'train':
                self.optimizer.zero_grad()
                self.optimizer_dec.zero_grad()
                dec = torch.cat(dec_list,dim=1)

                #reinforce_loss
                #R = loss_bert.view(-1,1,1).repeat(1,self.Data[i]['decoder_length']-1,config['input_size'])#.view(batch_size,self.Data[i]['decoder_length'],config['input_size'])
                #R = (R - R.mean(dim = 0))**2 / (R.std() + 1e-6)
                #print(R.shape)
                reinforce_loss = torch.mean(-torch.mul(dec,loss_bert))

                if args.loss == 'nll':
                    train_loss = loss
                elif args.loss == 'bert':
                    train_loss = reinforce_loss
                elif args.loss == 'combine':
                    train_loss = config['alpha'] * reinforce_loss + (1.0 - config['alpha']) * loss
                elif args.loss == 'alternate':
                    if torch.rand(1) < args.toggle_loss:
                        train_loss = loss
                    else:
                        train_loss = reinforce_loss
                train_loss.backward()
                for O_ in self.Opts:
                    O_.step()

        if type_ == 'eval':
            logging.info(
                f"Train:   Loss_MLE_eval: {loss_mle_inf:.4f},  Loss_Bert_eval: {loss_bert_inf:.4f}\n")
            wandb.log({'Loss_MLE_eval': loss_mle_inf, 'Loss_Bert_eval': loss_bert_inf}, step=ep)
            return loss_mle_inf
        if type_ == 'train':
            logging.info(
                f"Train:   Loss_MLE_train: {loss_mle_inf:.4f},  Loss_MLE_train: {loss_bert_inf:.4f}\n")
            wandb.log({'Loss_MLE_train': loss_mle_inf, 'Loss_Bert_train': loss_bert_inf}, step=ep)


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
    elif args.dataset == 'framse' and args.type == 'test':
        Data_test = FramesGraphDataset(suffix='test')

    Data_train.setBatchSize(config['batch_size'])

    Model = Seq2Seq(config)
    if args.reload:
        checkpoint = torch.load(os.path.join(saved_models, config['id'] + '_' + str(args.start_epoch)))
        Model.load_state_dict(checkpoint['model_state_dict'])
        config = checkpoint['config']
        wandb.init(project=config["wandb_project"], resume=config['id'])

    wandb.init(project=config["wandb_project"], name=config['id'], id=config['id'], allow_val_change=True)
    wandb.config.update(config)
    wandb.watch(Model)

    if args.type == 'train':
        Data_valid.setBatchSize(config['batch_size'])
        for epoch in range(config['epoch'], config['num_epochs']):
            logging.info(str(epoch) + '/' + str(config['num_epochs']))
            Model.modelrun(Data=Data_train, type_='train', total_step=Data_train.num_batches, ep=epoch,
                           sample_saver=None)
            config['epoch'] += 1
            torch.save({'model_State_dict': Model.state_dict(), 'config': config}, os.path.join(saved_models, config['id'] + '_-1'))
            if config['save_every_epoch']:
                torch.save({'model_State_dict': Model.state_dict(), 'config': config}, os.path.join(saved_models, config['id'] + '_' + str(epoch)))
            loss_mle_valid = Model.modelrun(Data=Data_valid, type_='eval', total_step=Data_valid.num_batches, ep=epoch, sample_saver=None)
            if loss_mle_valid < config['best_mle_valid']:
                config['best_mle_valid'] = loss_mle_valid
                wandb.config.update({'best_mle_valid':loss_mle_valid}, allow_val_change=True)
                torch.save({'model_State_dict': Model.state_dict(), 'config': config},
                           os.path.join(saved_models, config['id'] + '_best_mle_valid'))
                #save the best model to wandb
                torch.save({'model_State_dict': Model.state_dict(), 'config': config},
                           os.path.join(wandb.run.dir, config['id'] + '_best_mle_valid'))

    elif args.type == 'valid':
        if args.dont_use_best:
            sample_saver_valid = open(samples_fname + "_valid_" + config['id'] + '_' + str(args.start_epoch) + '.txt', 'w')
            sample_saver_valid = open(samples_fname + "_valid_" + config['id'] + '_' + str(args.start_epoch) + '.txt', 'a')
            # sample_saver_train = open(samples_fname + "_train_" + config['id'] + '_' + str(args.start_epoch), 'w')
            # sample_saver_train = open(samples_fname + "_train_" + config['id'] + '_' + str(args.start_epoch), 'a')
            checkpoint = torch.load(os.path.join(saved_models, config['id'] + '_' + str(args.start_epoch)))
        else:
            sample_saver_valid = open(samples_fname + "_valid_" + config['id'] + '_best_model.txt', 'w')
            sample_saver_valid = open(samples_fname + "_valid_" + config['id'] + '_best_model.txt', 'a')
            # sample_saver_train = open(samples_fname + "_train_" + config['id'] + '_best_model.txt', 'w')
            # sample_saver_train = open(samples_fname + "_train_" + config['id'] + '_best_model.txt', 'a')
            checkpoint = torch.load(os.path.join(saved_models, config['id'] + '_best_mle_valid'))
        Model.load_state_dict(checkpoint['model_state_dict'])
        # Model.modelrun(Data=Data_train, type_='valid', total_step=Data_valid.num_batches, ep=0,sample_saver=sample_saver_train)
        Model.modelrun(Data=Data_valid, type_='valid', total_step=Data_valid.num_batches, ep=0,sample_saver=sample_saver_valid)

    elif args.type == 'test':
        if args.dont_use_best:
            sample_saver_test = open(samples_fname+"_test_"+config['id'] + '_' + str(args.start_epoch) + '.txt','w')
            sample_saver_test = open(samples_fname+"_test_"+config['id'] + '_' + str(args.start_epoch) + '.txt','a')
            checkpoint = torch.load(os.path.join(saved_models, config['id'] + '_' + str(args.start_epoch)))
        else:
            sample_saver_test = open(samples_fname + "_test_" + config['id'] + '_best_model.txt', 'w')
            sample_saver_test = open(samples_fname + "_test_" + config['id'] + '_best_model.txt', 'a')
            checkpoint = torch.load(os.path.join(saved_models, config['id'] + '_best_mle_valid'))
        Model.load_state_dict(checkpoint['model_state_dict'])
        Model.modelrun(Data=Data_test, type_='valid', total_step=Data_test.num_batches, ep=0,sample_saver=sample_saver_test)
