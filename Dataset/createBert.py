from Frames_data_iterator import FramesGraphDataset
from WoZ_data_iterator import WoZGraphDataset
from bert_embedding import BertEmbedding

import argparse

bert = BertEmbedding()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default = 'frames')
args = parser.parse_args()

if args.dataset == 'frames':
    f = open('frames_embedding.txt','w')
    D = FramesGraphDataset()
else:
    f = open('mwoz_embedding.txt','w')
    D = WoZGraphDataset()

for i in range(D.vlen):
    print(i,'/',D.vlen)
    st = D.Vocab_inv[i]+'<del>'
    if i < 4:
        st += '<del>'.join([str(0.0)]*768)
    else:
        emb = bert([D.Vocab_inv[i]])
        st+= '<del>'.join([str(w) for w in emb[0][1][0].tolist()])
    f.write(st+'\n')
f.close()
