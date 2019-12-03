import os
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.meteor_score import meteor_score
import math
import argparse
import _pickle as cPickle
from rouge import Rouge
import logging

bleu_met = nltk.translate.bleu_score.sentence_bleu
parser = argparse.ArgumentParser()
parser.add_argument('--sample', type = str)
parser.add_argument('--increment',type = int, default = 4)
parser.add_argument('--topk',type = int, default = 2)
args = parser.parse_args()

R = Rouge()

def Metrics(file_loc):
    rouge_p = []
    rouge_r = []
    rouge_f = []
    bleu = []
    fp = open(file_loc)
    D = fp.readlines()
    r_pre = [0.0]*args.topk
    r_rec = [0.0]*args.topk
    r_f1 = [0.0]*args.topk
    sent_bleu = [0.0]*args.topk
    meteor_s = [0.0]*args.topk
    cnt_ = 0
    i=0
    while i<len(D):
        beam = []
        tar = D[i+args.topk+1].split()[2:]
        for j in range(1,args.topk+1):
            beam += [D[i+j].split()[2:]]
        if '<eos>' in tar:
            ind_tar = tar.index('<eos>')
        else:
            ind_tar = -1
        ind_mod = []
        for mod in beam:
            if '<eos>' in mod:
                ind_mod += [mod.index('<eos>')]
            else:
                ind_mod += [-1]
        for k in range(args.topk):
            r_scores = R.get_scores(' '.join(beam[k][:ind_mod[k]]),' '.join(tar[:ind_tar]))
            sent_bleu[k] += bleu_met([beam[k][:ind_mod[k]]],tar[:ind_tar],(0.5,0.5))
            meteor_s[k] += meteor_score([' '.join(beam[k][:ind_mod[k]])],' '.join(tar[:ind_tar]))
            r_pre[k] += r_scores[0]['rouge-l']['p']
            r_rec[k] += r_scores[0]['rouge-l']['r']
            r_f1[k] += r_scores[0]['rouge-l']['f']
        i+=args.increment+args.topk-1
        cnt_+=1
    score_dict = {}
    for k in range(args.topk):
        meteor_s[k] /= float(cnt_)
        r_pre[k] /= float(cnt_)
        r_rec[k] /= float(cnt_)
        r_f1[k] /= float(cnt_)
        sent_bleu[k]/=float(cnt_)
        score_dict.update({str(k+1):{'METEOR':meteor_s[k],'BLEU': sent_bleu[k],'F1': r_f1[k], 'Recall': r_rec[k], 'Precision': r_pre[k]}})

    return score_dict

if __name__ == '__main__':
    mets = Metrics(args.sample)
    print(mets)
    logging.info(mets)
