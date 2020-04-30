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

# requires csv with headers alpha values, epochs, meteor
def createData(dataset): # alpha->exp mapping {1:'1E-1',2:'1E-3'}
    mapdict = {2:'Baseline', 1: '1E-3', 0: '1E-2',3: '1E-1', 4: '1E0', 5:'1E1', 6:'1E2'}#{2:'Baseline', 1:'1E-3',4:'1E0'}#{2:'Baseline', 1: '1E-3', 0: '1E-2',3: '1E-1', 4: '1E0', 5:'1E1', 6:'1E2'}
    seeds = [100,101,102,103,104]
    fieldnames=['alpha','epoch','meteor']
    target = open("results.csv", "w")
    writer = csv.DictWriter(target, fieldnames=fieldnames)
    writer.writerow(dict(zip(fieldnames, fieldnames)))
    for k,v in mapdict.items():
        print('Now gathering info from ',v)
        for s in seeds:
            f = open('../Results/' + dataset + '/exp_' + str(k) + '_seed_' + str(s) + '/logs.txt')
            ep = 1
            for line in f:
                line = line.split()
                if line !=[]:
                    if 'Valid' in line[0]:
                        writer.writerow(dict([
                        ('alpha',v),
                        ('epoch',str(ep)),
                        ('meteor',line[-1])]))
                        ep+=1
    target.close()

def createGraph(yrange=[5,20], filename = 'results.csv'):
    plt.ylim(yrange[0],yrange[1])
    sns.lineplot(x = 'epoch', y ='meteor', hue = 'alpha',data = pd.read_csv(filename))
    plt.savefig('plot.png')

if __name__ == '__main__':
    if not os.path.exists('results.csv'):
        createData(args.dataset)
    createGraph()

