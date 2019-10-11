#Iterator of Babi-graph dataset with torch
from __future__ import print_function, division
import os
import torch
import _pickle as cPickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

class bAbIGraphDataset(Dataset):
    def __init__(self, Data_dir = '/home/ml/pparth2/Documents/GraphDial/CodeBase/Dataset/',pklfile = 'Data.pkl', suffix = 'train',batch_size = 1):
        #assert task_id > 0 and task_id < 21
        self.Data = cPickle.load(open(Data_dir + 'Dataset1_' + suffix + '_' + str(task)+'.pkl','rb'))
        self.Vocab = cPickle.load(open(Data_dir+ 'Vocab1_' + str(task) + '_train.pkl','rb'))
        self.Edges = cPickle.load(open(Data_dir+ 'Edges1_' + str(task) + '_train.pkl','rb'))
        self.len_data = len(self.Data)
        self.vlen = len(self.Vocab)
        self.elen = len(self.Edges)
        self.batch_size = batch_size

    def __len__(self):
        return self.len_data

    def setBatchSize(self,batch_size = 32):
        self.batch_size = batch_size

    def __getitem__(self, idx):
        ''' idt for task id
        idc for context id
        ids for sentence id'''
        def getGraph(edge_list):
            graph = []
            assert len(edge_list)>0
            for edge_ in edge_list:
                if edge_['from'] in self.Vocab and edge_['to'] in self.Vocab:
                    vertices = [self.Vocab[edge_[nodes]] for nodes in ['from','to']]
                    edge = [self.Edges[edge_[nodes]] for nodes in ['type']]
                    graph+=[(vertices[0],edge[0],vertices[1])]
            return set(graph)
        D = self.Data[idx]
        vertices_ = []
        edges_ = []
        sentences_ = []
        questions = []
        answers = []
        decoder_mask = []
        ind = 0
        q_ind = []
        for k,_ in D.items():
            ind = k-1
            if 'sentence' in D[k].keys():
                indices = [self.Vocab[x] if x in self.Vocab else 2 for x in D[k]['sentence']]
                #decoder_mask_ = [1.]*(len(indices)+2)+[0.]*(10-len(indices))
                indices = [0]+indices+[1]*(12-len(indices))
                one_hot_sent = np.zeros((len(indices),self.vlen),dtype = np.float32)
                one_hot_sent[np.arange(len(indices)),indices] = 1.
                c_one_hot_vert = np.zeros((1,self.vlen),dtype = np.float32).sum(axis = 0)
                c_one_hot_edge = np.zeros((1,self.elen),dtype = np.float32)
                if len(D[k]['c_graph']) and D[k]['c_graph'][0]['from'] in self.Vocab and D[k]['c_graph'][0]['to'] in self.Vocab:
                    vertices = [self.Vocab[D[k]['c_graph'][0][nodes]] for nodes in ['from','to']]
                    c_one_hot_vert = np.zeros((len(vertices),self.vlen),dtype = np.float32)
                    c_one_hot_vert[np.arange(len(vertices)),vertices] = 1.
                    c_one_hot_vert = np.sum(c_one_hot_vert, axis = 0)
                    edge = [self.Edges[D[k]['c_graph'][0][nodes]] for nodes in ['type']]
                    assert len(edge) == 1
                    c_one_hot_edge = np.zeros((len(edge),self.elen),dtype = np.float32)
                    c_one_hot_edge[np.arange(len(edge)),edge] = 1.
                vertices_ += [c_one_hot_vert]
                edges_ += [c_one_hot_edge[0,:]]
                sentences_ += [one_hot_sent]
            else:
                q_ind += [ind]
                indices_q = [self.Vocab[x] if x in self.Vocab else 2 for x in D[k]['question']]
                indices_q = [0]+indices_q+[1]*(12 - len(indices_q))
                one_hot_ques = np.zeros((len(indices_q),self.vlen),dtype = np.float32)
                one_hot_ques[np.arange(len(indices_q)),indices_q] = 1.
                indices_a = [self.Vocab[x] if x in self.Vocab else 2 for x in D[k]['fact_ind']]
                indices_a = [0]+indices_a+[1]*(12 - len(indices_a))
                one_hot_ans = np.zeros((len(indices_a),self.vlen),dtype = np.float32)
                one_hot_ans[np.arange(len(indices_a)),indices_a] = 1.
                questions+=[one_hot_ques]
                answers+=[one_hot_ans]
        assert len(edges_) == len(vertices_)
        assert len(vertices_) == len(sentences_)
        try:
            graph = getGraph(D[len(D)-1]['graph']['edges'])
        except:
            graph = getGraph(D[len(D)-2]['graph']['edges'])
        return {'graph': graph, 'edges': np.array(edges_),'batch_size': len(D), 'question_ind': q_ind,'vertices': np.array(vertices_), 'sentences': np.array(sentences_),'questions':questions, 'answers':answers}

    def getBatch(self,batch_id = 0):
        Data_ = []
        for ind in range(batch_id*self.batch_size,min(self.len_data,(batch_id+1)*self.batch_size)):
            Data_+=[self.__getitem__(ind)]
        return Data_
