import seaborn as sns
sns.set()
import pandas as pd
import csv
import os
import argparse
import matplotlib.pyplot as plt
import _pickle as pickle

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = 'Frames')
parser.add_argument('--graphparam',default='% unseen')
args = parser.parse_args()

def extractNgramsFromData(D,ngrams,type_='train',n=2,person='user'):
    if person == 'user':
        start = 0
    else:
        start = 1
    for _,dial in D.items():
        for i in range(start,len(dial['log']),2):
            for l in [dial['log'][i]['utterance'][:-1]]:
                for j in range(n,len(l)+1):
                    if tuple(l[j-n:j]) in ngrams:
                        ngrams[tuple(l[j-n:j])][type_]+=1
                    else:
                        ngrams.update({tuple(l[j-n:j]):{'train': 0, 'valid':0 ,'test':0}})
                        ngrams[tuple(l[j-n:j])][type_]+=1
    return set(ngrams.keys())

def getNgramsfromSample(file,n=1):
    ngrams = {}
    fp = open(file)
    D = fp.readlines()
    i = 0
    repeats = 0
    while i<len(D):
        #tar = D[i+2].split()[1:]
        mod = D[i+1].split()[1:]
        if '<eos>' in mod:
            ind_mod = mod.index('<eos>')
        else:
            ind_mod = -1
        #print(mod[:ind_mod],len(ngrams))
        for l in [mod[:ind_mod]]:
             for j in range(n,len(l)+1):
                 print(tuple(l[i-n:i]))
                 if len(set(l[j-n:j])) < len(tuple(l[j-n:j])):
                     repeats+=1
                 if tuple(l[j-n:j]) in ngrams:
                     ngrams[tuple(l[j-n:j])] +=1
                 else:
                     ngrams.update({tuple(l[j-n:j]):1})
        i+=4
    total_ngrams = sum([v for k,v in ngrams.items()])+0.0001
    return set(ngrams.keys()), float(repeats)/total_ngrams


def extractStats(dataset):
    if dataset == 'Frames':
        folder = '../Dataset/Frames-dataset/'
    else:
        folder = '../Dataset/MULTIWOZ2/'
    ngrams = {}
    D = pickle.load(open(os.path.join(folder,'Dataset_train_'+dataset+'.pkl'),'rb'))
    ngrams = extractNgramsFromData(D, ngrams, type_='train', n=2, person = 'agent')
    mapdict = {20:'Baseline', 23: 'BERT', 30: 'fastText',31: 'GloVe'}#{2:'Baseline', 1:'1E-3',4:'1E0'}#{2:'Baseline', 1: '1E-3', 0: '1E-2',3: '1E-1', 4: '1E0', 5:'1E1', 6:'1E2'}
    seeds = [101,102,103]
    fieldnames=['LM Emb','epoch','word repeats','% unseen']
    target = open("quality_analysis_"+dataset+"_valid.csv", "w")
    writer = csv.DictWriter(target, fieldnames=fieldnames)
    writer.writerow(dict(zip(fieldnames, fieldnames)))
    for k,v in mapdict.items():
        print('Now gathering info from ',v)
        for s in seeds:
            print(s)
            if 'rames' in dataset:
                dataset = 'frames'
            else:
                dataset = 'mwoz'
            folder = '../Results/' + dataset + '/exp_' + str(k) + '_seed_' + str(s) + '/Samples'
            ep = 1
            for num in range(100):#os.listdir(folder):
                file = 'samples_valid_exp_'+str(k)+'_seed_'+str(s)+'_'+str(num)+'.txt'
                path = os.path.join(folder,file)
                if not os.path.exists(path):
                    break
                else:
                    bi_grams,word_repeats = getNgramsfromSample(path,n=2)
                    percent_unseen = (len(bi_grams) - len(bi_grams.intersection(ngrams)))/(len(bi_grams)+0.0001)
                    writer.writerow(dict([
                    ('LM Emb',v),
                    ('epoch',str(ep)),
                    ('word repeats',str(word_repeats)),
                    ('% unseen',str(percent_unseen))])
                    )
                    ep+=1
    target.close()

def createGraph(yrange=[5,20], filename = 'ngrams_valid.csv', graphparam = '% unigram'):
    #plt.ylim(yrange[0],yrange[1])
    sns.lineplot(x = 'epoch', y =graphparam, hue = 'LM Emb',data = pd.read_csv(filename))
    plt.savefig('plot_'+graphparam+'.png')

if __name__ == '__main__':
    if not os.path.exists("quality_analysis_"+args.dataset+"_valid.csv"):
        extractStats(args.dataset)
    createGraph(filename = "quality_analysis_"+args.dataset+"_valid.csv", graphparam = args.graphparam)
