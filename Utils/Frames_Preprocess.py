import os
import numpy as np
import json
import _pickle as cPickle
import argparse
from nltk.tokenize import word_tokenize
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--type',default = 'train')
args = parser.parse_args()
data_directory = '../Dataset/Frames-dataset/'
fp = open(data_directory+'frames.json')
raw_data = json.load(fp)
e_in = 0
vin = 3
if args.type == 'train':
    Vocab = {}
    Vocab['<go>'] = 0
    Vocab['<eos>'] = 1
    Vocab['<unk>'] = 2
    E_vocab = {}
else:
    Vocab = cPickle.load(open(data_directory+'Vocab.pkl','rb'))
    E_vocab = cPickle.load(open(data_directory+'Edges.pkl','rb'))

def get_data(file_):
    global Vocab, E_vocab, vin, e_in
    Dialog = {}
    D_global = {}
    max_ut_len = 0
    if len(f['turns'])%2 == 0:
        dial_len = len(f['turns'])
    else:
        dial_len = len(f['turns']) - 1
    context = []
    for i in range(dial_len):
        text = f['turns'][i]['text']
        text_ = []
        for token in word_tokenize(text):
            if token.lower().strip() not in Vocab and args.type == 'train':
                Vocab.update({token.lower().strip():vin})
                vin+=1
            text_+=[token.lower().strip()]
        D_local = {}
        def getfromframe(info):
            global Vocab, E_vocab, vin, e_in
            graph = {}
            keys = list(info.keys())
            for k in keys:
                if k not in E_vocab and args.type == 'train':
                    E_vocab.update({str(k):e_in})
                    e_in+=1
                if 'list' not in str(type(info[k][0]['val'])) and str(info[k][0]['val']) not in Vocab and args.type == 'train':
                    Vocab.update({str(info[k][0]['val']):vin})
                    vin+=1
                graph.update({k:str(info[k][0]['val'])})
            return graph
        if i%2 == 0:
            for kv_pair in f['turns'][i]['labels']['acts']:
                for kv in kv_pair['args']:
                    if 'val' in kv:
                        if 'list' not in str(type(kv['val'])):
                            D_local.update({kv['key']:str(kv['val'])})
                            if str(kv['key']) not in E_vocab and args.type == 'train':
                                E_vocab.update({str(kv['key']):e_in})
                                e_in+=1
                            if str(kv['val']) not in Vocab and args.type == 'train':
                                Vocab.update({str(kv['val']):vin})
                                vin+=1
                        else:
                            frame_id = kv['val'][-1]['frame'] -1
                            D_local = getfromframe(f['turns'][i]['labels']['frames'][frame_id]['info'])
            context += text_
            text_ = deepcopy(context)
        if i == 0:
            Dialog.update({'log':{i:{'utterance':text_, 'local_graph': D_local}}})
        else:
            Dialog['log'].update({i:{'utterance':text_, 'local_graph': D_local}})
    return Dialog


if __name__ == '__main__':
    Dataset = {}
    in_ = 0
    L = []
    def get_users_for_fold(fold):
        folds = {'U21E41CQP': 1,
                 'U23KPC9QV': 1,
                 'U21RP4FCY': 2,
                 'U22HTHYNP': 3,
                 'U22K1SX9N': 4,
                 'U231PNNA3': 5,
                 'U23KR88NT': 6,
                 'U24V2QUKC': 7,
                 'U260BGVS6': 8,
                 'U2709166N': 9,
                 'U2AMZ8TLK': 10}

        if fold < 0:
            ret = [k for k, v in folds.items() if v != -fold]
        else:
            ret = [k for k, v in folds.items() if v == fold]
        return ret

    test_users = get_users_for_fold(2)
    train_users = get_users_for_fold(-2)

    train_dialogues = [d for d in raw_data if d['user_id'] in train_users]
    test_dialogues = [d for d in raw_data if d['user_id'] in test_users]

    if args.type == 'train':
        dialogues = train_dialogues
    else:
        dialogues = test_dialogues

    for f in dialogues:
        i_D = get_data(f)
        if len(i_D) > 0:
            if len(i_D['log']) > 0:
                Dataset.update({in_: i_D})
                in_+=1
        print(in_,'/',len(dialogues))
    if args.type == 'train':
        cPickle.dump(E_vocab,open(data_directory + '/Edges.pkl','wb'))
        cPickle.dump(Vocab,open(data_directory + '/Vocab.pkl','wb'))
    cPickle.dump(Dataset,open(data_directory + '/Dataset_'+args.type+ '_Frames.pkl','wb'))
