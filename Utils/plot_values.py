import seaborn as sns
sns.set()
import pandas as pd
import csv
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = 'frames')
parser.add_argument('--graphparam', default = 'METEOR')
args = parser.parse_args()

# requires csv with headers alpha values, epochs, meteor
def createData(dataset): # alpha->exp mapping {1:'1E-1',2:'1E-3'}
    mapdict = {20:'Baseline', 22: '1E-2',23: '1E-1', 24: '1E0', 25:'1E1'}#{2:'Baseline', 1:'1E-3',4:'1E0'}#{2:'Baseline', 1: '1E-3', 0: '1E-2',3: '1E-1', 4: '1E0', 5:'1E1', 6:'1E2'}
    seeds = [100,101,103,104]
    fieldnames=['alpha','epoch','NLL Train','BERT Train','METEOR','BLEU','NLL Eval','BERT Eval']
    target = open("results_valid.csv", "w")
    writer = csv.DictWriter(target, fieldnames=fieldnames)
    writer.writerow(dict(zip(fieldnames, fieldnames)))
    for k,v in mapdict.items():
        print('Now gathering info from ',v)
        for s in seeds:
            f = open('../Results/' + dataset + '/exp_' + str(k) + '_seed_' + str(s) + '/logs.txt')
            ep_v = 1
            ep_t = 1
            for line in f:
                line = line.split(',')
                if line !=['\n']:
                    if 'Valid' in line[1]:
                        writer.writerow(dict([
                        ('alpha',v),
                        ('epoch',str(ep_v)),
                        ('BLEU',line[-1].split()[-1]),
                        ('METEOR',line[-2].split()[-1]),
                        ('NLL Eval',line[-4].split()[-1]),
                        ('BERT Eval',line[-3].split()[-1])])
                        )
                        ep_v+=1
                    if 'Train' in line[1]:
                        writer.writerow(dict([
                        ('alpha',v),
                        ('epoch',str(ep_t)),
                        ('NLL Train',str(ep_t)),
                        ('BERT Train',str(ep_t))])
                        )
                        ep_t+=1
    target.close()

def createGraph(yrange=[5,20], filename = 'results_valid.csv', graphparam = 'METEOR'):
    #plt.ylim(yrange[0],yrange[1])
    sns.lineplot(x = 'epoch', y =graphparam, hue = 'alpha',data = pd.read_csv(filename))
    plt.savefig('plot_'+graphparam+'.png')

if __name__ == '__main__':
    if not os.path.exists('results_valid.csv'):
        createData(args.dataset)
    createGraph(graphparam = args.graphparam)
