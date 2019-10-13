import os
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import math
import argparse
import _pickle as cPickle
from rouge import Rouge

bleu_met = nltk.translate.bleu_score.sentence_bleu
parser = argparse.ArgumentParser()
parser.add_argument('--sample', type = str)
parser.add_argument('--increment',type = int, default = 4)
args = parser.parse_args()

R = Rouge()

def Metrics(file_loc):
    rouge_p = []
    rouge_r = []
    rouge_f = []
    bleu = []
    fp = open(file_loc)
    D = fp.readlines()
    r_pre = 0.0
    r_rec = 0.0
    r_f1 = 0.0
    sent_bleu = 0.0
    cnt_ = 0
    i=0
    while i<len(D):
        tar = D[i+2].split()[2:]
        mod = D[i+1].split()[1:]
        ind = tar.index('<eos>')
        r_scores = R.get_scores(' '.join(mod[:ind]),' '.join(tar[:ind]))
        sent_bleu += bleu_met([mod],tar,(0.5,0.5))
        r_pre += r_scores[0]['rouge-l']['p']
        r_rec += r_scores[0]['rouge-l']['r']
        r_f1 += r_scores[0]['rouge-l']['f']
        i+=args.increment
        cnt_+=1
    return {'BLEU': sent_bleu/float(cnt_),'F1': r_f1/float(cnt_), 'Recall': r_rec/float(cnt_), 'Precision': r_pre/float(cnt_)}

if __name__ == '__main__':
    mets = Metrics(args.sample)
    print(mets)
