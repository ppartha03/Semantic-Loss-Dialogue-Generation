#Iterator of Babi-graph dataset with torch
from __future__ import print_function, division
import os
import torch
import _pickle as cPickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

max_length = 100
class WoZGraphDataset(Dataset):
    def __init__(self, Data_dir = '../Dataset/MULTIWOZ2/',suffix = 'train',batch_size = 1):
        #assert task_id > 0 and task_id < 21
        self.Data_dir = Data_dir
        self.Data = cPickle.load(open(Data_dir+'Dataset_' + suffix + '_WoZ.pkl','rb'))
        self.Vocab = cPickle.load(open(Data_dir+'Vocab.pkl','rb'))
        self.Vocab_inv = {}
        for k,v in self.Vocab.items():
            self.Vocab_inv.update({v:k})
        self.Edges = cPickle.load(open(Data_dir+'Edges.pkl','rb'))
        self.len_data = len(self.Data)
        self.vlen = len(self.Vocab)
        self.elen = len(self.Edges)
        self.batch_size = batch_size
        self.num_batches = int(self.len_data/self.batch_size)

    def __len__(self):
        return self.len_data

    def setBatchSize(self,batch_size = 32):
        self.batch_size = batch_size
        self.num_batches = int(self.len_data/self.batch_size)

    def __getitem__(self, start_ind):
        ''' idt for task id
        idc for context id
        ids for sentence id'''
        #template, graph, dialogue, message
        def getGraph(edge_dict):
            graph = []
            print(edge_dict)
            #assert len(edge_dict)>0
            for k,v in edge_dict.items():
                vertices = [self.Vocab[v.split()[0].lower()] if v.split()[0].strip().lower() in self.Vocab else 2]
                edge = [self.Edges[k.strip()] if k.strip() in self.Edges else 2]
                graph+=[(vertices[0],edge[0])]
            return set(graph)

        vertices_bow = []
        edges_bow = []
        sentences_ = []
        user_utt = []
        machine_utt = []
        answers = []
        decoder_mask = []
        ind = 0
        for idx in range(start_ind*self.batch_size,min(start_ind*self.batch_size+self.batch_size,self.len_data)):
            D = self.Data[idx]
            logs = D['log']
            max_seq_len = 0
            max_encoder_len = 0
            for k,_ in logs.items():
                text = logs[k]['utterance']
                if k%2==0:
                    u_edges = []
                    u_vertices = []
                    user_graph = logs[k]['local_graph']
                    for edge,value in user_graph.items():
                        c_one_hot_vert = np.zeros((1,self.vlen),dtype = np.float32).sum(axis = 0)
                        c_one_hot_edge = np.zeros((1,self.elen),dtype = np.float32).sum(axis = 0)
                        if len(value.split()):
                            vertices = [self.Vocab[value.split()[0].strip().lower()] if value.split()[0].strip().lower() in self.Vocab else 2]
                        else:
                            vertices = [2]
                        c_one_hot_vert = np.zeros((len(vertices),self.vlen),dtype = np.float32)
                        c_one_hot_vert[np.arange(len(vertices)),vertices] = 1.
                        c_one_hot_vert = np.sum(c_one_hot_vert, axis = 0)
                        if len(edge.strip()):
                            edge_ = [self.Edges[edge.strip()] if edge.strip() in self.Edges else 2]
                        else:
                            edge_ = [2]
                        c_one_hot_edge = np.zeros((len(edge_),self.elen),dtype = np.float32)
                        c_one_hot_edge[np.arange(len(edge_)),edge_] = 1.
                        u_vertices += [c_one_hot_vert]
                        u_edges += [c_one_hot_edge[0,:]]
                    if len(u_edges) > 1:
                        edges_bow += [np.sum(np.array(u_edges), axis = 0).reshape(-1)]
                        vertices_bow += [np.sum(np.array(u_vertices), axis = 0).reshape(-1)]
                    else:
                        vertices_bow += [np.zeros((1,self.vlen),dtype = np.float32).sum(axis = 0)]
                        edges_bow += [np.zeros((1,self.elen),dtype = np.float32).sum(axis = 0)]
                if len(text) < max_length-2:
                    indices = [self.Vocab[text[x].strip()] if text[x] in self.Vocab else 2 for x in range(len(text))]
                else:
                    indices = [self.Vocab[text[x].strip()] if text[x] in self.Vocab else 2 for x in range(max_length-2)]
                #decoder_mask_ = [1.]*(len(indices)+2)+[0.]*(10-len(indices))
                indices = [0]+indices+[1]*(max_length-len(indices))
                one_hot_sent = np.zeros((len(indices),self.vlen),dtype = np.float32)
                one_hot_sent[np.arange(len(indices)),indices] = 1.
                #print(k)
                if k%2==0:
                    user_utt +=[one_hot_sent]
                    if len(text) > max_encoder_len:
                        max_encoder_len = len(text)
                else:
                    machine_utt += [one_hot_sent]
                    if len(text) > max_seq_len:
                        max_seq_len = len(text)
            #print(len(user_utt),len(machine_utt))
        if start_ind*self.batch_size < self.len_data:
            return {'encoder_length':min(max_length,max_encoder_len),'decoder_length': min(max_length,max_seq_len),'input':np.array(user_utt), 'vertices':np.array(vertices_bow), 'edges': np.array(edges_bow), 'target': np.array(machine_utt)}
        else:
            raise Exception('x should be less than {}. The value of index was: {}'.format(self.len_data, start_ind))
    def getBatch(self,batch_id = 0):
        Data = self.__getitem__(ind)
        return Data
