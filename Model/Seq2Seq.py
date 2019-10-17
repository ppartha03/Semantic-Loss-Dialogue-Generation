from RNN import EncoderRNN, DecoderRNN, Q_predictor
import torch.nn.functional as F
import sys
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import numpy as np
import argparse
import time
import os
sys.path.append('../Utils/')

parser = argparse.ArgumentParser()
#parser.add_argument('--task')
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--dataset', type=str, default="frames")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--type', type=str, default='train')
parser.add_argument('--teacher_forcing', type=float, default=0.1)
parser.add_argument('--encoder_learning_rate', type=float, default=0.004)
parser.add_argument('--decoder_learning_rate', type=float, default=0.004)
parser.add_argument('--data_path', type=str, default="../Dataset")
parser.add_argument('--save_path', type=str, default="..")
args = parser.parse_args()

sys.path.append(args.data_path)
from WoZ_data_iterator import WoZGraphDataset
from Frames_data_iterator import FramesGraphDataset
from Bert_util import load_embeddings, bert_metric


if not os.path.exists(args.save_path + '/Results/Seq2Seq/'+args.dataset+'/Samples/'):
    os.makedirs(args.save_path + '/Results/Seq2Seq/'+args.dataset+'/Samples/')

if not os.path.exists(args.save_path + '/Seq2Seq/Saved_Models/'):
    os.makedirs(args.save_path + '/Seq2Seq/Saved_Models')

fname = args.save_path + '/Results/Seq2Seq/'+args.dataset+'/Log.txt'
samples_fname = args.save_path + '/Results/Seq2Seq/'+args.dataset+'/Samples/Sample_'

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
config['save_path'] = args.save_path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
config['num_epochs'] = 10000
config['decoder_learning_rate'] = args.decoder_learning_rate
config['encoder_learning_rate'] = args.encoder_learning_rate
config['batch_size'] = args.batch_size
config['num_edges'] = config['data'].elen
config['num_vertices'] = config['data'].vlen


config['weights'] = [1,0] + [1]*(config['input_size'] - 2)
# 1 => account for loss
# 0 => mask the token
# embedding matrix Nxd


class Seq2Seq(nn.Module):
    def __init__(self, config, embeddings):
        super(Seq2Seq, self).__init__()
        #Encoder_model can be s2s or hred
        self.Data = config['data']
        self.config = config
        self.Encoder = EncoderRNN(self.config['input_size'], self.config['hidden_size'], self.config['num_layers'],num_edges = self.Data.elen, num_vertices = self.Data.vlen,).to(device)
        self.Decoder = DecoderRNN(self.config['hidden_size'], self.config['output_size'], self.config['input_size'], self.config['num_layers']).to(device)
        self.Bert_embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
        # Loss and optimizer
        self.criterion = nn.NLLLoss(weight = torch.from_numpy(np.array(config['weights'])).float()).to(device)
        # criterion_2 = nn.CrossEntropyLoss().to(device)

        self.optimizer = torch.optim.RMSprop(self.Encoder.parameters(), lr = config['encoder_learning_rate'],alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        self.optimizer_dec = torch.optim.Adam(self.Decoder.parameters(), lr =config['decoder_learning_rate'])
        self.Opts = [self.optimizer, self.optimizer_dec]

        self.writer = SummaryWriter()

    def modelrun(self, Data = '', type_ = 'train', total_step = 200, ep = 0, sample_saver = '', saver = ''):
        loss_inf = 0.
        self.Data = Data
        self.sample_saver = sample_saver

        for i in range(total_step):
            seq_loss_a = 0.
            batch_size = self.Data[i]['input'].shape[0]
            hidden_enc = (torch.zeros(self.config['num_layers'], batch_size, self.config['hidden_size'], device=device), torch.zeros(self.config['num_layers'], batch_size, self.config['hidden_size'], device=device))

            input_ = torch.from_numpy(self.Data[i]['input']).to(device).view(batch_size,self.config['sequence_length'],self.config['num_vertices'])
            decoder_input = torch.from_numpy(self.Data[i]['target']).to(device).view(batch_size,self.config['sequence_length'],self.config['num_vertices'])
            edges_t = torch.from_numpy(self.Data[i]['edges']).to(device).view(batch_size,self.config['num_edges'])
            vertices_t = torch.from_numpy(self.Data[i]['vertices']).to(device).view(batch_size, self.config['num_vertices'])

            # if type_ == 'valid':
            response_ = []
            context_ = []
            target_response = []

            for di in range(self.config['sequence_length']):
                o_v, o_e, hidden_enc, out = self.Encoder(input_[:,di,:].view(-1,1,self.config['input_size']),hidden_enc)

            decoder_hidden = hidden_enc

            decoder_input_ = decoder_input[:,0,:]

            for di in range(self.config['sequence_length']-1):
                decoder_output, decoder_hidden = self.Decoder(decoder_input_.view(-1,1,self.config['input_size']), decoder_hidden)
                if np.random.rand() > self.config['teacher_forcing'] and type_ == 'train':
                    decoder_input_ = decoder_output.view(-1,1,self.config['input_size'])
                else:
                    decoder_input_ = decoder_input[:,di+1,:].view(-1,1,self.config['input_size'])
                seq_loss_a += self.criterion(input = decoder_output[:,-1,:], target = torch.max(decoder_input[:,di+1,:], dim = 1)[-1])
                # if type_ == 'valid':
                response_ = response_ + [torch.argmax(decoder_output.view(batch_size,-1),dim =1).view(-1,1)]
                context_ = context_ + [torch.argmax(input_[:,di,:].view(batch_size,-1),dim =1).view(-1,1)]
                target_response = target_response + [torch.argmax(decoder_input[:,di,:].view(batch_size,-1),dim =1).view(-1,1)]
            con = torch.cat(context_, dim=1)
            res = torch.cat(response_, dim=1)
            tar = torch.cat(target_response, dim=1)

            loss = seq_loss_a/batch_size/config['sequence_length']
            loss_inf += seq_loss_a.item()/batch_size/config['sequence_length']
            bert_loss = bert_metric(self.Bert_embedding(res), self.Bert_embedding(tar))

            if type_ == 'valid':
                for c_index in range(con.shape[0]):
                    c = ' '.join([self.Data.Vocab_inv[idx.item()] for idx in con[c_index]])
                    r = ' '.join([self.Data.Vocab_inv[idx.item()] for idx in res[c_index]])
                    t = ' '.join([self.Data.Vocab_inv[idx.item()] for idx in tar[c_index]])
                    self.sample_saver.write('Context: '+ c + '\n' + 'Model_Response: ' + r + '\n' + 'Target: ' + t + '\n\n')
            if type_ == 'train':
                self.optimizer.zero_grad()
                self.optimizer_dec.zero_grad()
                loss.backward()
                for O_ in self.Opts:
                    O_.step()

        if type_ == 'eval':
            self.writer.add_scalar('Loss/LM_Loss_eval', loss_inf, ep)
        if type == 'train':
            self.writer.add_scalar('Loss/LM_Loss_train', loss_inf, ep)



if __name__ == '__main__':
    if args.dataset == 'mwoz':
        Data_train = WoZGraphDataset()
        Data_valid = WoZGraphDataset(suffix = 'valid')
    else:
        Data_train = FramesGraphDataset()
        Data_valid = FramesGraphDataset(suffix = 'valid')
    Data_train.setBatchSize(config['batch_size'])

    words, embs = load_embeddings(args)
    Model = Seq2Seq(config, embs)
    if args.type == 'train':
        Data_valid.setBatchSize(config['batch_size'])
        for epoch in range(config['num_epochs']):
            print(epoch,'/',config['num_epochs'])
            saver = open(fname,'a')
            Model.modelrun(Data = Data_train, type_ = 'train', total_step = Data_train.num_batches , ep = epoch,sample_saver = None,saver = saver)
            torch.save(Model.state_dict(), config['save_path'] + '/Seq2Seq/Saved_Models/Seq2Seq_'+str(args.hidden_size)+ '_'+args.dataset+'_'+str(args.batch_size))
            Model.modelrun(Data = Data_valid, type_ = 'eval', total_step = Data_valid.num_batches , ep = epoch,sample_saver = None,saver = saver)
    elif args.type == 'valid':
            sample_saver = open(samples_fname+'valid.txt','w')
            sample_saver = open(samples_fname+'valid.txt','a')
            Model.load_state_dict(torch.load(config['save_path'] + '/Seq2Seq/Saved_Models/Seq2Seq_'+str(args.hidden_size)+ '_'+args.dataset+'_'+str(args.batch_size)))
            Model.modelrun(Data = Data_valid, type_ = 'valid', total_step = Data_valid.num_batches, ep = 0,sample_saver = sample_saver,saver = saver)
