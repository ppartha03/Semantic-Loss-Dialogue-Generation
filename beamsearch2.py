from Model.RNN import EncoderRNN, AttnDecoderRNN
from Utils.Eval_metric import getscores
from Utils.Bert_util import Load_embeddings, Sem_reward, Mask_sentence, Posteos_mask, create_id, BeamSearchNode, getTopK
from Dataset_utils.Frames_data_iterator import FramesGraphDataset
from Dataset_utils.WoZ_data_iterator import WoZGraphDataset
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import logging
import wandb
import random
from torch.distributions.categorical import Categorical
from queue import PriorityQueue

parser = argparse.ArgumentParser()
# parser.add_argument('--task')
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--dataset', type=str, default="frames", choices=["frames", "mwoz"])
# nll, sem, combine, alternate
parser.add_argument('--loss', type=str, default='combine')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--toggle_loss', type=float, default=0.0)
parser.add_argument('--teacher_forcing', type=float, default=1.0)
parser.add_argument('--change_nll_mask', action='store_true')
parser.add_argument('--results_path', type=str, default='.')
parser.add_argument('--encoder_learning_rate', type=float, default=0.004)
parser.add_argument('--decoder_learning_rate', type=float, default=0.004)
parser.add_argument('--output_dropout', type=float, default=0.0)
parser.add_argument('--data_path', type=str, default="./Dataset")
parser.add_argument('--save_every_epoch', action='store_true')
parser.add_argument('--reward_baseline_n_steps', type=int, default=20,
                    help="Number of steps taken in calculating the running average of the reward as baseline, "
                         "use 0 for no reward baseline")
parser.add_argument('--topk', type=int, default=5)
parser.add_argument('--exp_id', type=int, default=0)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=100)
# calculate sentence embedding using "mean" or "sum" of word embeddings
parser.add_argument('--sentence_embedding', type=str, default='mean')
#Use None for not logging using wandb
parser.add_argument('--wandb_project', type=str, default=None)
# which model to use for validation/test, 'best_mle' or 'best_combined',
# 'best_meteor'
parser.add_argument('--validation_model', type=str, default='best_meteor',
                    choices=["best_mle", "best_combined_loss", "best_meteor", "best_bleu", "last"])
parser.add_argument(
    '--embeddings',
    type=str,
    default='bert')  # glove, word2vec, bert
args = parser.parse_args()
config = vars(args)
config["last_n_rewards"] = []
if config["wandb_project"] == 'None' or config["wandb_project"] == 'none':
    config["wandb_project"] = None
config["run_id"] = "exp_" + str(args.exp_id) + "_seed_" + str(args.seed)
config["wandb_id"] = str(random.randint(1e7, 1e8))
sys.path.append(args.data_path)

result_path = os.path.join(args.results_path, "Results", args.dataset, config["run_id"])
if not os.path.exists(result_path):
    os.makedirs(result_path)

saved_models = os.path.join(result_path, 'Saved_Models')
if not os.path.exists(saved_models):
    os.makedirs(saved_models)

samples_path = os.path.join(result_path, 'Samples')
if not os.path.exists(samples_path):
    os.makedirs(samples_path)

config_fname = os.path.join(result_path, 'config.txt')
logfile_name = os.path.join(result_path,'logs.txt')
f = open(config_fname,"w")
f.write(str(config))
f.close()

# Hyper-parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config['device'] = device

if args.dataset == 'mwoz':
    Data_train = WoZGraphDataset(
    Data_dir=args.data_path + '/MULTIWOZ2/')
    Data_valid = WoZGraphDataset(
        Data_dir=args.data_path + '/MULTIWOZ2/', suffix='valid')
    Data_test = WoZGraphDataset(
        Data_dir=args.data_path + '/MULTIWOZ2/', suffix='test')
else:
    Data_train = FramesGraphDataset(
        Data_dir=args.data_path + '/Frames-dataset/')
    Data_valid = FramesGraphDataset(
        Data_dir=args.data_path + '/Frames-dataset/', suffix='valid')
    Data_test = FramesGraphDataset(
        Data_dir=args.data_path + '/Frames-dataset/', suffix='test')


# Hyper-parameters
config['sequence_length'] = 101
config['input_size'] = Data_train.vlen
config['num_layers'] = 1
config['output_size'] = Data_train.vlen

config['best_mle_valid'] = 10000
config['best_mle_valid_epoch'] = 0
config['best_combined_loss'] = 10000
config['best_combined_loss_epoch'] = 0
config['best_meteor'] = 0
config['best_meteor_epoch'] = 0
config['best_bleu'] = 0
config['best_bleu_epoch'] = 0

# Create model id
config['mask_special_indices'] = np.array([0, 1, 1, 0])
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
        # Encoder_model can be s2s or hred
        self.config = config
        self.mask = np.hstack(
            [config['mask_special_indices'],
             np.ones(config['input_size'] - len(config['mask_special_indices']))])
        self.Encoder = EncoderRNN(self.config['input_size'], self.config['hidden_size'], self.config['num_layers']).to(
            self.config['device'])
        self.Decoder = AttnDecoderRNN(self.config['hidden_size'], self.config['output_size'], self.config['num_layers'],
                                      self.config['sequence_length']).to(self.config['device'])
        _, self.weights = Load_embeddings(config['dataset'], config['embeddings'], args.data_path)
        self.Sem_embedding = nn.Embedding.from_pretrained(
            self.weights, freeze=True).to(self.config['device'])
        # Loss and optimizer
        self.criterion = nn.NLLLoss(
            weight=torch.from_numpy(
                self.mask).float()).to(self.config['device'])
        # criterion_2 = nn.CrossEntropyLoss().to(self.config['device'])

        self.optimizer = torch.optim.RMSprop(
            self.Encoder.parameters(),
            lr=config['encoder_learning_rate'],
            alpha=0.99,
            eps=1e-08,
            weight_decay=0,
            momentum=0,
            centered=False)
        self.optimizer_dec = torch.optim.Adam(
            self.Decoder.parameters(),
            lr=config['decoder_learning_rate'])
        self.Opts = [self.optimizer, self.optimizer_dec]

    def modelrun(self, Data, type_='train', total_step=200, ep=0, sample_saver=None):
        loss_mle_inf = 0.
        sem_reward_inf = 0.
        loss_reinforce_inf = 0.
        train_loss_inf = 0.
        config = self.config

        self.sample_saver = sample_saver
        count_examples = 0.
        for i in range(total_step):
            batch_size = Data[i]['input'].shape[0]
            count_examples += batch_size
            hidden_enc = (
                torch.zeros(
                    config['num_layers'],
                    batch_size,
                    config['hidden_size'],
                    device=config['device']),
                torch.zeros(
                    config['num_layers'],
                    batch_size,
                    config['hidden_size'],
                    device=config['device']))

            input_ = torch.from_numpy(
                Data[i]['input']).to(
                config['device']).to(
                torch.int64).view(
                batch_size,
                config['sequence_length'])
            decoder_input = torch.from_numpy(
                Data[i]['target']).to(
                config['device']).to(
                torch.int64).view(
                batch_size,
                config['sequence_length'])

            response_greedy = []
            response_sampled = []
            context_ = []

            encoder_outputs = torch.zeros(
                config['sequence_length'],
                batch_size,
                config['hidden_size'],
                device=config['device'])

            self.optimizer.zero_grad()
            self.optimizer_dec.zero_grad()
            for di in range(Data[i]['encoder_length'] + 1):
                out, hidden_enc = self.Encoder(input_[:, di], hidden_enc)
                context_ = context_ + [input_[:, di].unsqueeze(1)]
                encoder_outputs[di] = (
                    hidden_enc[0] + hidden_enc[1]).view(batch_size, -1)

            decoder_hidden = hidden_enc

            # Apply output dropout fpr sem loss (generate random masks)
            mask = torch.from_numpy(self.mask).to(device).bool()
            if type_ == 'train' and config['output_dropout'] > 0:
                weight_random = np.random.random(
                    len(mask) - len(config['mask_special_indices'])) > config['output_dropout']
                mask_numpy = np.hstack(
                    [config['mask_special_indices'], weight_random.astype(int)])
                mask = torch.from_numpy(mask_numpy).to(device).bool()
                # Apply output dropout for MLE loss
                if config['change_nll_mask']:
                    self.criterion = nn.NLLLoss(weight=torch.from_numpy(
                        mask_numpy).float()).to(device)

            target_response = [decoder_input[:, di + 1].unsqueeze(1) for
                               di in range(Data[i]['decoder_length'] + 1)]
            tar = torch.cat(target_response, dim=1)
            loss = 0.
            reinforce_loss = 0.

            #beam decoder: here
            topk = args.topk
            decoded_batch = []
            for b_ind in range(Data[i]['input'].shape[0]):
                decoder_hidden = (hidden_enc[0][:,b_ind, :],hidden_enc[1][:,b_ind, :])
                decoder_input_ = decoder_input[b_ind,0]

                node = BeamSearchNode(decoder_hidden, None, torch.argmax(decoder_input_).item(), 0., 1)
                nodes = PriorityQueue()
                endnodes = []
                number_required = min((topk + 1), topk - len(endnodes))
                nodes.put((-node.eval(), node))
                qsize = 1
                while True:
                    # give up when decoding takes too long
                    if qsize > 20000: break

                    score, n = nodes.get()
                    decoder_input_ = n.wordid
                    decoder_hidden = n.h
                    if ( n.wordid == 1 and n.prevNode != None) or n.leng == 20:
                        endnodes.append((score, n))
                        # if we reached maximum # of sentences required
                        if len(endnodes) >= number_required:
                            break
                        else:
                            continue

                    decoder_output, decoder_hidden_o = self.Decoder(decoder_input_, decoder_hidden, encoder_outputs)
                    log_prob, indexes = torch.topk(decoder_output.view(-1), args.beam_width)
                    nextnodes = []
                    logging.info(str(indexes))
                    for new_k in range(args.beam_width):
                        decoded_t = indexes[new_k].item()
#                        print(decoded_t)
                        logging.info(str(decoded_t))
                        log_p = log_prob[new_k].item()
                        one_hot_dt = torch.zeros(1,self.config['input_size']).float().to(self.config['device'])
                        one_hot_dt[0,decoded_t] = 1.
                        node = BeamSearchNode(decoder_hidden_o, n, decoded_t, n.logp + log_p, n.leng + 1)
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

            if config['loss'] == 'nll':
                train_loss = loss
            elif config['loss'] == 'sem':
                train_loss = reinforce_loss
            elif config['loss'] == 'combine':
                train_loss = config['alpha'] * \
                    reinforce_loss + loss  # changed loss
            elif config['loss'] == 'alternate':
                if torch.rand(1) < args.toggle_loss:
                    train_loss = loss
                else:
                    train_loss = reinforce_loss
            train_loss_inf += train_loss.item() / total_step

            if type_ !='train':
                for c_index in range(con.shape[0]):
                    c = ' '.join([Data.Vocab_inv[idx.item()] for idx in con[c_index]])
                    t_list = [Data.Vocab_inv[idx.item()] for idx in tar[c_index]]
                    t = ' '.join(t_list)
                    r = ''
                    for beam_ind in range(args.topk):
                        res = decoded_batch[c_index][beam_ind]
                        res_str = ' '.join([self.Data.Vocab_inv[idx] for idx in res])
                        r += 'Model_Response_'+str(beam_ind)+': '+res_str+'\n'
                    self.sample_saver.write('Context: '+ c + '\n' + r + 'Target: ' + t + '\n\n')

            if type_ != 'train':
                for c_index in range(con.shape[0]):
                    c = ' '.join([Data.Vocab_inv[idx.item()]
                                  for idx in con[c_index]])
                    t_list = [Data.Vocab_inv[idx.item()]
                              for idx in tar[c_index]]
                    t = ' '.join(t_list)
#                    cnt += 1
                    r_list = [Data.Vocab_inv[idx.item()]
                              for idx in res[c_index]]
                    r = ' '.join(r_list)
                    self.sample_saver.write(
                        'Context: ' +
                        c +
                        '\n' +
                        'Model_Response: ' +
                        r +
                        '\n' +
                        'Target: ' +
                        t +
                        '\n\n')

            if type_ == 'train':

                train_loss.backward()
                for O_ in self.Opts:
                    O_.step()

        if type_ == 'eval':
            # meteor_score_valid = meteor_score_valid / cnt * 100
            self.sample_saver.close()
            valid_scores = getscores(sample_saver.name)
            meteor_score_valid = valid_scores['METEOR'] * 100.
            bleu_score_valid = valid_scores['BLEU'] * 100.

            logging.info(
                f"Valid:   Loss_MLE_eval: {loss_mle_inf:.4f},  Sem_reward_eval: {sem_reward_inf:.4f}, "
                f"'meteor_score': {meteor_score_valid:.2f},"
                f"'bleu_score': {bleu_score_valid:.2f}\n")
            if config["wandb_project"] is not None:
                wandb.log({'Loss_MLE_eval': loss_mle_inf, 'Sem_reward_eval': sem_reward_inf,
                           'train_loss_eval': train_loss_inf, 'reinforce_loss_eval': loss_reinforce_inf,
                           'meteor_score': meteor_score_valid, 'bleu_score': bleu_score_valid, 'global_step':ep})
            return loss_mle_inf, train_loss_inf, meteor_score_valid, bleu_score_valid
        if type_ == 'train':
            logging.info(
                f"Train:   Loss_MLE_train: {loss_mle_inf:.4f},  Sem_reward_train: {sem_reward_inf:.4f}\n")
            if config["wandb_project"] is not None:
                wandb.log({'Loss_MLE_train': loss_mle_inf, 'Sem_reward_train': sem_reward_inf,
                           'train_loss_train': train_loss_inf, 'reinforce_loss_train': loss_reinforce_inf,
                           'global_step':ep})


if __name__ == '__main__':
    logging.basicConfig(filename=logfile_name,
                        filemode='a+',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    if os.path.exists(os.path.join(saved_models, config['run_id'] + '_last')):
        logging.info(
            f"Resuming training of model " + os.path.join(saved_models, config['run_id'] + '_last'))
        checkpoint = torch.load(os.path.join(saved_models, config['run_id'] + '_last'))
        config = checkpoint['config']
        config["device"] = device
        Model = Seq2Seq(config)
        Model.load_state_dict(checkpoint['model_State_dict'])
        logging.info(str(config))
        if config["wandb_project"] is not None:
            wandb.init(project=config["wandb_project"], resume=config["wandb_id"], allow_val_change=True)
    else:
        Model = Seq2Seq(config)
        torch.save({'model_State_dict': Model.state_dict(), 'config': config},
                   os.path.join(saved_models, config['run_id'] + '_last'))
        if config["wandb_project"] is not None:
            wandb.init(project=config["wandb_project"], name=config['run_id'], id=config["wandb_id"], allow_val_change=True)

    # Train
    Data_train.setBatchSize(config['batch_size'])
    if config["wandb_project"] is not None:
        wandb.config.update(config, allow_val_change=True)
        wandb.watch(Model)
    logging.info(f"using {config['device']}\n")

    Data_valid.setBatchSize(config['batch_size'])
    for epoch in range(config['epoch'], args.num_epochs):
        logging.info(str(epoch) + '/' + str(args.num_epochs))
        Model.modelrun(Data=Data_train, type_='train', total_step=Data_train.num_batches, ep=epoch)
        config['epoch'] += 1
        if config["wandb_project"] is not None:
            wandb.config.update({'epoch': config['epoch']}, allow_val_change=True)
        torch.save({'model_State_dict': Model.state_dict(
        ), 'config': config}, os.path.join(saved_models, config['run_id'] + '_last'))
        if config['save_every_epoch']:
            torch.save({'model_State_dict': Model.state_dict(), 'config': config}, os.path.join(
                saved_models, config['run_id'] + '_' + str(epoch)))

        sample_saver_eval = open(os.path.join(
            samples_path,
            "beam_"+args.topk+"samples_valid_" +
            config['run_id'] + '_' +
            str(epoch) +
            '.txt'),
            'w')

        loss_mle_valid, combined_loss_valid, meteor_valid, bleu_valid = \
            Model.modelrun(Data=Data_valid, type_='eval', total_step=Data_valid.num_batches,
                           ep=epoch, sample_saver=sample_saver_eval)


        if bleu_valid > config['best_bleu']:
            config['best_bleu'] = bleu_valid
            config['best_bleu_epoch'] = epoch
            if config["wandb_project"] is not None:
                wandb.config.update({'best_bleu':bleu_valid}, allow_val_change=True)
                wandb.config.update({'best_bleu_epoch': epoch}, allow_val_change=True)
            torch.save({'model_State_dict': Model.state_dict(), 'config': config},
                       os.path.join(saved_models, config['run_id'] + '_best_bleu'))
            # save the best model to wandb
            if config["wandb_project"] is not None:
                torch.save({'model_State_dict': Model.state_dict(), 'config': config},
                os.path.join(wandb.run.dir, config['run_id'] + '_best_bleu'))
        if meteor_valid > config['best_meteor']:
            config['best_meteor'] = meteor_valid
            config['best_meteor_epoch'] = epoch
            if config["wandb_project"] is not None:
                wandb.config.update({'best_meteor':meteor_valid}, allow_val_change=True)
                wandb.config.update({'best_meteor_epoch': epoch}, allow_val_change=True)
            torch.save({'model_State_dict': Model.state_dict(), 'config': config},
                       os.path.join(saved_models, config['run_id'] + '_best_meteor'))
            # save the best model to wandb
            if config["wandb_project"] is not None:
                torch.save({'model_State_dict': Model.state_dict(), 'config': config},
                os.path.join(wandb.run.dir, config['run_id'] + '_best_meteor'))
        if loss_mle_valid < config['best_mle_valid']:
            config['best_mle_valid'] = loss_mle_valid
            config['best_mle_valid_epoch'] = epoch
            if config["wandb_project"] is not None:
                wandb.config.update({'best_mle_valid':loss_mle_valid}, allow_val_change=True)
                wandb.config.update({'best_mle_valid_epoch': epoch}, allow_val_change=True)
            torch.save({'model_State_dict': Model.state_dict(), 'config': config},
                       os.path.join(saved_models, config['run_id'] + '_best_mle'))
            # save the best model to wandb
            if config["wandb_project"] is not None:
                torch.save({'model_State_dict': Model.state_dict(), 'config': config},
                os.path.join(wandb.run.dir, config['run_id'] +
                '_best_mle'
                ''))
        if combined_loss_valid < config['best_combined_loss']:
            config['best_combined_loss'] = combined_loss_valid
            config['best_combined_loss_epoch'] = epoch
            if config["wandb_project"] is not None:
                wandb.config.update({'best_combined_loss':combined_loss_valid}, allow_val_change=True)
                wandb.config.update({'best_combined_loss_epoch': epoch}, allow_val_change=True)
            torch.save({'model_State_dict': Model.state_dict(), 'config': config},
                       os.path.join(saved_models, config['run_id'] + '_best_combined_loss'))
            # save the best model to wandb
            if config["wandb_project"] is not None:
                torch.save({'model_State_dict': Model.state_dict(), 'config': config},
                os.path.join(wandb.run.dir, config['run_id'] + '_best_combined_loss'))

    if config["wandb_project"] is not None:
        torch.save({'model_State_dict': Model.state_dict(), 'config': config},
            os.path.join(wandb.run.dir, config['run_id'] + '_last'))

    # Validation and test
    checkpoint = torch.load(
        os.path.join(
            saved_models,
            config['run_id'] + '_' +
            args.validation_model))

    Model.load_state_dict(checkpoint['model_State_dict'])

    sample_saver_valid = open(os.path.join(
        samples_path,
        "beam_"+str(args.topk)+"samples_valid_"+
        config['run_id'] + '_' +
        args.validation_model +
        '.txt'),
        'w')
    logging.info(f"{args.validation_model} model validation results:")
    loss_mle_valid, combined_loss_valid, meteor_valid, bleu_valid = \
        Model.modelrun(Data=Data_valid, type_='eval', total_step=Data_valid.num_batches, ep=0,
                       sample_saver=sample_saver_valid)

    sample_saver_test = open(os.path.join(
        samples_path,
        "beam_"+args.topk+"samples_test_" +
        config['run_id'] + '_' +
        args.validation_model +
        '.txt'),
        'w')
    logging.info(f"{args.validation_model} model test results:")
    loss_mle_test, combined_loss_test, meteor_test, bleu_test = \
        Model.modelrun(Data=Data_test, type_='eval', total_step=Data_test.num_batches, ep=0,
                       sample_saver=sample_saver_test)
