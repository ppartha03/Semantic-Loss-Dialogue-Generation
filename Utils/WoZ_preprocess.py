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
data_directory = '../Dataset/MULTIWOZ2/'
fp = open(data_directory+'data.json')
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

def recur(level, key, root, D = {}, D_local = {}):
    global Vocab, E_vocab, vin, e_in
    if 'dict' not in str(type(level)):
      return True
    else:
      keys = list(level.keys())
      for k in keys:
          key = root+'#'+k
          if 'dict' not in str(type(level[k])):
              if 'list' not in str(type(level[k])) and level[k] != '' and len(level[k]) != 0 and level[k] != 'not mentioned':
                  if key not in D:
                      D_local.update({key:level[k]})
                      D.update({key:level[k]})
                      if key.strip() not in E_vocab and args.type == 'train':
                          E_vocab.update({key.strip():e_in})
                          e_in+=1
          recur(level[k],k,key,D,D_local)
          key = root
    return D_local,D

def get_data(file_):
    global Vocab, E_vocab, vin, e_in
    Dialog = {}
    D_global = {}
    context = []
    for i in range(len(raw_data[file_]['log'])):
        text = raw_data[file_]['log'][i]['text']
        text_ = []
        for token in word_tokenize(text):
            if token.lower().strip() not in Vocab and args.type == 'train':
                Vocab.update({token.lower().strip():vin})
                vin+=1
            text_+=[token.lower().strip()]
        D_local = {}
        if i%2 == 0:
            D_local, D_global = recur(raw_data[file_]['log'][i+1]['metadata'],'','',D_global,D_local)
            context += text_
            text_ = deepcopy(context)
        else:
            D_local = {}
        if i == 0:
            Dialog.update({'log':{i:{'utterance':text_, 'local_graph': D_local, 'global': D_global}}})
        else:
            Dialog['log'].update({i:{'utterance':text_, 'local_graph': D_local, 'global': D_global}})
    return Dialog


if __name__ == '__main__':
    Files = list(raw_data.keys())
    Dataset = {}
    in_ = 0
    f_dev = open(data_directory + 'testListFile.json')
    dev_files = [line.strip() for line in f_dev.readlines()]
    f_test = open(data_directory + 'valListFile.json')
    test_files = [line.strip() for line in f_test.readlines()]
    train_files = [f for f in Files if f not in dev_files and f not in test_files]
    print(len(train_files),len(test_files))

    if args.type == 'train':
        dialogue_files = train_files
    else:
        dialogue_files = test_files

    for f in dialogue_files:
        i_D = get_data(f)
        if len(i_D) > 0:
            if len(i_D['log']) > 0:
                Dataset.update({in_: i_D})
                in_+=1
        print(in_,'/',len(dialogue_files))

    if args.type == 'train':
        cPickle.dump(E_vocab,open(data_directory + '/Edges.pkl','wb'))
        cPickle.dump(Vocab,open(data_directory + '/Vocab.pkl','wb'))
    cPickle.dump(Dataset,open(data_directory + '/Dataset_'+args.type+ '_WoZ.pkl','wb'))
