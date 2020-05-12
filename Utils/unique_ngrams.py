import seaborn as sns
sns.set()
import pandas as pd
import csv
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = 'frames')
args = parser.parse_args()

def getNgrams(file,n=1):
    ngrams = {}
    fp = open(file)
    D = fp.readlines()
    while i<len(D):
        #tar = D[i+2].split()[1:]
        mod = D[i+1].split()[1:]
        for l in [mod]:
             for i in range(n,len(l)+1):
                 if tuple(l[i-n:i]) in ngrams:
                     ngrams[tuple(l[i-n:i])] +=1
                 else:
                     ngrams.update({tuple(l[i-n:i]):1})
    return float(len(ngrams))/float(sum([v for k,v in ngrams.items()]))

def distinctNgrams(dataset): # alpha->exp mapping {1:'1E-1',2:'1E-3'}
    mapdict = {20:'Baseline', 21: '1E-3', 22: '1E-2',23: '1E-1', 24: '1E0', 25:'1E1'}#{2:'Baseline', 1:'1E-3',4:'1E0'}#{2:'Baseline', 1: '1E-3', 0: '1E-2',3: '1E-1', 4: '1E0', 5:'1E1', 6:'1E2'}
    seeds = [100,101,103,104]
    fieldnames=['alpha','epoch','% unigram','% bigram']
    target = open("ngrams_valid.csv", "w")
    writer = csv.DictWriter(target, fieldnames=fieldnames)
    writer.writerow(dict(zip(fieldnames, fieldnames)))
    for k,v in mapdict.items():
        print('Now gathering info from ',v)
        for s in seeds:
            folder = open('../Results/' + dataset + '/exp_' + str(k) + '_seed_' + str(s) + '/Samples')
            ep = 1
            for file in os.listdir(folder):
                path = os.path.join(folder,file)
                one_grams = getNgrams(path,n=1)
                bi_grams = getNgrams(path,n=2)
                writer.writerow(dict([
                ('alpha',v),
                ('epoch',str(ep)),
                ('% unigram',str(one_grams)),
                ('% bigram',str(bi_grams))
                )

def createGraph(yrange=[5,20], filename = 'ngrams_valid.csv', graphparam = gp):
    #plt.ylim(yrange[0],yrange[1])
    sns.lineplot(x = 'epoch', y =gp, hue = 'alpha',data = pd.read_csv(filename))
    plt.savefig('plot_'+graphparam.png')

if __name__ == '__main__':
    if not os.path.exists('ngrams_valid.csv'):
        distinctNgrams(args.dataset)
    createGraph(graphparam = args.graphparam)
