from Model.RNN import EncoderRNN, DecoderRNN, Q_predictor, AttnDecoderRNN
from Utils.Eval_metric import meteor
from Utils.Bert_util import Load_embeddings, Bert_loss, Mask_sentence, Posteos_mask, create_id
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

parser = argparse.ArgumentParser()
# parser.add_argument('--task')
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--dataset', type=str, default="frames", choices=["frames", "mwoz"])
# nll, sem, combine, alternate
parser.add_argument('--loss', type=str, default='combine')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--toggle_loss', type=float, default=0.5)
parser.add_argument('--teacher_forcing', type=float, default=0.1)
parser.add_argument('--change_nll_mask', action='store_true')
parser.add_argument('--results_path', type=str, default='.')
parser.add_argument('--encoder_learning_rate', type=float, default=0.004)
parser.add_argument('--decoder_learning_rate', type=float, default=0.004)
parser.add_argument('--output_dropout', type=float, default=0.0)
parser.add_argument('--data_path', type=str, default="./Dataset")
parser.add_argument('--save_every_epoch', action='store_true')
parser.add_argument('--exp_id', type=int, default=0)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=100)
# calculate sentence embedding using "mean" or "sum" of bert embeddings
parser.add_argument('--sentence_embedding', type=str, default='mean')
# if true, don't apply the mask before generating the Bert sentence (allow
# the model to generate masked tokens, and then mask them during the
# embedding calculation)
parser.add_argument('--no_prebert_mask', action='store_true')
#Use None for not logging using wandb
parser.add_argument('--wandb_project', type=str, default=None)
# which model to use for validation/test, 'best_mle' or 'best_combined',
# 'best_meteor'
parser.add_argument('--validation_model', type=str, default='best_meteor',
                    choices=["best_mle", "best_combined_loss", "best_meteor", "last"])
parser.add_argument(
    '--embeddings',
    type=str,
    default='bert')  # glove, word2vec, bert
args = parser.parse_args()
config = vars(args)
if config["wandb_project"] == 'None' or config["wandb_project"] == 'none':
    config["wandb_project"] = None
config["run_id"] = "exp_" + str(args.exp_id) + "_seed_" + str(args.seed)
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
    Data_dir=config['data_path'] + '/MULTIWOZ2/')
    Data_valid = WoZGraphDataset(
        Data_dir=config['data_path'] + '/MULTIWOZ2/', suffix='valid')
    Data_test = WoZGraphDataset(
        Data_dir=config['data_path'] + '/MULTIWOZ2/', suffix='test')
else:
    Data_train = FramesGraphDataset(
        Data_dir=config['data_path'] + '/Frames-dataset/')
    Data_valid = FramesGraphDataset(
        Data_dir=config['data_path'] + '/Frames-dataset/', suffix='valid')
    Data_test = FramesGraphDataset(
        Data_dir=config['data_path'] + '/Frames-dataset/', suffix='test')


# Hyper-parameters
config['sequence_length'] = 101
config['input_size'] = Data_train.vlen
config['num_layers'] = 1
config['output_size'] = Data_train.vlen

config['best_mle_valid'] = 10000
config['best_mle_valid_epoch'] = 0
config['best_combined_loss'] = 10000
config['best_combined_loss_epoch'] = 0
config['meteor_valid'] = 0
config['meteor_valid_epoch'] = 0

# Create model id
config['mask'] = np.hstack(
    [np.array([1, 1, 1, 0]), np.ones(config['input_size'] - 4)])
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
        self.Encoder = EncoderRNN(
            self.config['input_size'],
            self.config['hidden_size'],
            self.config['num_layers']).to(
            self.config['device'])
        self.Decoder = AttnDecoderRNN(
            self.config['hidden_size'],
            self.config['output_size'],
            self.config['num_layers'],
            self.config['sequence_length']).to(self.config['device'])
        _, self.weights = Load_embeddings(config['dataset'], config['embeddings'], config['data_path'])
        self.Bert_embedding = nn.Embedding.from_pretrained(
            self.weights, freeze=True).to(self.config['device'])
        # Loss and optimizer
        self.criterion = nn.NLLLoss(
            weight=torch.from_numpy(
                config['mask']).float()).to(
            self.config['device'])
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
        loss_bert_inf = 0.
        loss_reinforce_inf = 0.
        train_loss_inf = 0.

        self.sample_saver = sample_saver
        count_examples = 0.
        for i in range(total_step):
            seq_loss_a = 0.
            batch_size = Data[i]['input'].shape[0]
            count_examples += batch_size
            hidden_enc = (
                torch.zeros(
                    self.config['num_layers'],
                    batch_size,
                    self.config['hidden_size'],
                    device=self.config['device']),
                torch.zeros(
                    self.config['num_layers'],
                    batch_size,
                    self.config['hidden_size'],
                    device=self.config['device']))

            input_ = torch.from_numpy(
                Data[i]['input']).to(
                self.config['device']).view(
                batch_size,
                self.config['sequence_length'],
                self.config['input_size'])
            decoder_input = torch.from_numpy(
                Data[i]['target']).to(
                self.config['device']).view(
                batch_size,
                self.config['sequence_length'],
                self.config['input_size'])

            # if type_ == 'valid':
            response_ = []
            context_ = []
            target_response = []
            response_premasked = []  # the response generated by choosing only unmasked words
            # Generate random masks
            encoder_outputs = torch.zeros(
                self.config['sequence_length'],
                batch_size,
                self.config['hidden_size'],
                device=self.config['device'])

            self.optimizer.zero_grad()
            self.optimizer_dec.zero_grad()
            for di in range(Data[i]['encoder_length']):
                out, hidden_enc = self.Encoder(input_[:, di, :], hidden_enc)
                context_ = context_ + \
                    [torch.argmax(input_[:, di, :].view(batch_size, -1), dim=1).view(-1, 1)]
                encoder_outputs[di] = (
                    hidden_enc[0] + hidden_enc[1]).view(batch_size, -1)

            decoder_hidden = hidden_enc

            decoder_input_ = decoder_input[:, 0, :]
            dec_list = []
            mask = torch.from_numpy(config['mask']).to(device).bool()

            if type_ == 'train' and config['output_dropout'] > 0:
                weight_random = np.random.random(
                    len(config['mask']) - 4) > config['output_dropout']
                config['mask'] = np.hstack(
                    [config['mask'][:4], weight_random.astype(int)])
                mask = torch.from_numpy(config['mask']).to(device).bool()
                if config['change_nll_mask']:
                    self.criterion = nn.NLLLoss(
                        weight=torch.from_numpy(
                            config['mask']).float()).to(device)

            for di in range(Data[i]['decoder_length'] - 1):
                decoder_output, decoder_hidden, _ = self.Decoder(
                    decoder_input_, decoder_hidden, encoder_outputs)

                if np.random.rand() < self.config['teacher_forcing'] and type_ == 'train':
                    decoder_input_ = decoder_input[:, di + 1, :]
                else:
                    decoder_input_ = decoder_output.view(-1, self.config['input_size'])

                seq_loss_a += self.criterion(input=decoder_output[:, -1, :], target=torch.max(
                    decoder_input[:, di + 1, :], dim=1)[-1])
                dec_list += [decoder_output.view(-1,1, self.config['input_size'])]
                target_response = target_response + \
                    [torch.argmax(decoder_input[:, di, :].view(batch_size, -1), dim=1).view(-1, 1)]
                response_ = response_ + \
                    [torch.argmax(decoder_output.view(batch_size, -1), dim=1).view(-1, 1)]

                if type_ == 'train':
                    response_premasked = response_premasked + \
                        [torch.argmax(decoder_output.view(batch_size, -1).masked_fill(~mask, -10**6), dim=1).view(-1, 1)]
                else:
                    response_premasked = response_
            con = torch.cat(context_, dim=1)
            res = torch.cat(response_, dim=1)
            res_premasked = torch.cat(response_premasked, dim=1)
            tar = torch.cat(target_response, dim=1)

            loss = seq_loss_a / batch_size / Data[i]['decoder_length']
            # seq_loss_a.item()/batch_size/Data[i]['decoder_length']
            loss_mle_inf += loss.item() / total_step

            res_premasked = Posteos_mask(res_premasked, config)
            loss_bert = Bert_loss(
                self.Bert_embedding(res_premasked),
                self.Bert_embedding(tar),
                config['sentence_embedding'])

            loss_bert_inf += loss_bert.item() / total_step

            dec = torch.cat(dec_list, dim=1)
            # Extracting only the exact words: greedy
            words = torch.max(dec,dim=2)[0]
            print(words.shape, loss_bert.shape)
            reinforce_loss = torch.mean(-torch.matmul(words, loss_bert))
            loss_reinforce_inf += reinforce_loss.item() / total_step

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
                    # if '<eos>' in t_list:
                    #     ind_tar = t_list.index('<eos>')
                    # else:
                    #     ind_tar = -1
                    # if '<eos>' in r_list:
                    #     ind_mod = r_list.index('<eos>')
                    # else:
                    #     ind_mod = -1
                    # meteor_score_valid += meteor_score(' '.join(r_list[:ind_mod]),' '.join(t_list[1:ind_tar]))
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
            meteor_score_valid = meteor(sample_saver.name) * 100.
            logging.info(
                f"Valid:   Loss_MLE_eval: {loss_mle_inf:.4f},  Loss_Bert_eval: {loss_bert_inf:.4f}, "
                f"'meteor_score': {meteor_score_valid:.2f}\n")
            if config["wandb_project"] is not None:
                wandb.log({'Loss_MLE_eval': loss_mle_inf, 'Loss_Bert_eval': loss_bert_inf,
                           'train_loss_eval': train_loss_inf, 'reinforce_loss_eval': loss_reinforce_inf,
                           'meteor_score': meteor_score_valid, 'global_step':ep})
            return loss_mle_inf, train_loss_inf, meteor_score_valid
        if type_ == 'train':
            logging.info(
                f"Train:   Loss_MLE_train: {loss_mle_inf:.4f},  Loss_Bert_train: {loss_bert_inf:.4f}\n")
            if config["wandb_project"] is not None:
                wandb.log({'Loss_MLE_train': loss_mle_inf, 'Loss_Bert_train': loss_bert_inf,
                           'train_loss_train': train_loss_inf, 'reinforce_loss_train': loss_reinforce_inf,
                           'global_step':ep})


if __name__ == '__main__':
    logging.basicConfig(filename=logfile_name,
                        filemode='a',
                        level=logging.INFO)

    Model = Seq2Seq(config)
    if os.path.exists(os.path.join(saved_models, config['run_id'] + '_last')):
        checkpoint = torch.load(os.path.join(saved_models, config['run_id'] + '_last'))
        Model.load_state_dict(checkpoint['model_State_dict'])
        config = checkpoint['config']
        config["device"] = device
        if config["wandb_project"] is not None:
            wandb.init(project=config["wandb_project"], resume=config['run_id'], allow_val_change=True)
    else:
        torch.save({'model_State_dict': Model.state_dict(), 'config': config},
                   os.path.join(saved_models, config['run_id'] + '_last'))
        if config["wandb_project"] is not None:
            wandb.init(project=config["wandb_project"], name=config['run_id'], id=config['run_id'], allow_val_change=True)

    # Train
    Data_train.setBatchSize(config['batch_size'])
    if config["wandb_project"] is not None:
        wandb.config.update(config, allow_val_change=True)
        wandb.watch(Model)
    logging.info(f"using {config['device']}\n")

    Data_valid.setBatchSize(config['batch_size'])
    for epoch in range(config['epoch'], config['num_epochs']):
        logging.info(str(epoch) + '/' + str(config['num_epochs']))
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
            "samples_valid_" +
            config['run_id'] + '_' +
            str(epoch) +
            '.txt'),
            'a+')
        loss_mle_valid, combined_loss_valid, meteor_valid = Model.modelrun(Data=Data_valid, type_='eval',
                                                                           total_step=Data_valid.num_batches,
                                                                           ep=epoch, sample_saver=sample_saver_eval)
        if meteor_valid > config['meteor_valid']:
            config['meteor_valid'] = meteor_valid
            if config["wandb_project"] is not None:
                wandb.config.update({'meteor_valid':meteor_valid}, allow_val_change=True)
                wandb.config.update({'meteor_valid_epoch': epoch}, allow_val_change=True)
            torch.save({'model_State_dict': Model.state_dict(), 'config': config},
                       os.path.join(saved_models, config['run_id'] + '_best_meteor'))
            # save the best model to wandb
            if config["wandb_project"] is not None:
                torch.save({'model_State_dict': Model.state_dict(), 'config': config},
                os.path.join(wandb.run.dir, config['run_id'] + '_best_meteor'))
        if loss_mle_valid < config['best_mle_valid']:
            config['best_mle_valid'] = loss_mle_valid
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
            config['run_id'] +
            args.validation_model))

    Model.load_state_dict(checkpoint['model_State_dict'])

    sample_saver_valid = open(os.path.join(
        samples_path,
        "samples_valid_" +
        config['run_id'] + '_' +
        args.validation_model +
        '.txt'),
        'a+')
    Model.modelrun(Data=Data_valid, type_='eval', total_step=Data_valid.num_batches, ep=0,
                   sample_saver=sample_saver_valid)

    sample_saver_test = open(os.path.join(
        samples_path,
        "samples_test_" +
        config['run_id'] + '_' +
        args.validation_model +
        '.txt'),
        'a+')
    Model.modelrun(Data=Data_test, type_='eval', total_step=Data_test.num_batches, ep=0,
                   sample_saver=sample_saver_test)
