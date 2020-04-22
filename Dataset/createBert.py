from Dataset_utils.Frames_data_iterator import FramesGraphDataset
from Dataset_utils.WoZ_data_iterator import WoZGraphDataset
from bert_embedding import BertEmbedding
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = 'frames')
parser.add_argument('--embeddings', default = 'bert', choices=["bert", "word2vec", "glove"])
args = parser.parse_args()

if args.dataset == 'frames':
    D = FramesGraphDataset()
else:
    D = WoZGraphDataset()

f = open(args.dataset+'_'+args.embeddings+'.txt','w')
m = 0
if args.embeddings == 'bert':
    bert = BertEmbedding()
    for i in range(D.vlen):
        print(i,'/',D.vlen)
        st = D.Vocab_inv[i]+'<del>'
        if i < 4:
            st += '<del>'.join([str(0.0)]*768)
        else:
            emb = bert([D.Vocab_inv[i]])
            st+= '<del>'.join([str(w) for w in emb[0][1][0].tolist()])
        f.write(st+'\n')

elif args.embeddings == 'word2vec':
    word2vec = {}
    fp = open('wiki-news-300d-1M.vec')
    for line in fp:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word2vec[word] = coefs
    fp.close()
    for i in range(D.vlen):
        print(i,'/',D.vlen)
        st = D.Vocab_inv[i]+'<del>'
        if i < 4:
            st += '<del>'.join([str(0.0)]*300)
        else:
            if D.Vocab_inv[i] in word2vec:
                emb = word2vec[D.Vocab_inv[i]]
            else:
                emb = np.random.rand(300)
                m+=1
            st+= '<del>'.join([str(w) for w in emb.tolist()])
        f.write(st+'\n')
    print('Missed values:',m)

elif args.embeddings == 'glove':
    glove_index = {}
    fp = open('glove.6B.300d.txt')
    for line in fp:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        glove_index[word] = coefs
    fp.close()
    for i in range(D.vlen):
        print(i,'/',D.vlen)
        st = D.Vocab_inv[i]+'<del>'
        if i < 4:
            st += '<del>'.join([str(0.0)]*300)
        else:
            if D.Vocab_inv[i] in glove_index:
                emb = glove_index[D.Vocab_inv[i]]
            else:
                emb = np.random.rand(300)
                m+=1
            st+= '<del>'.join([str(w) for w in emb.tolist()])
        f.write(st+'\n')
    print('Missed values:',m)
else:
    print('Check the args again!')
f.close()
