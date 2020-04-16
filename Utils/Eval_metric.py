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

def Metrics(file_loc, increment=4):
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
    meteor_s = 0.0
    cnt_ = 0
    i=0
    while i<len(D):
        tar = D[i+2].split()[2:]
        mod = D[i+1].split()[1:]

        if '<eos>' in tar:
            ind_tar = tar.index('<eos>')
        else:
            ind_tar = -1
        if '<eos>' in mod:
            ind_mod = mod.index('<eos>')
        else:
            ind_mod = -1
        r_scores = R.get_scores(' '.join(mod[:ind_mod]),' '.join(tar[:ind_tar]))
        sent_bleu += bleu_met([mod[:ind_mod]],tar[:ind_tar],(0.5,0.5))
        meteor_s += meteor_score([' '.join(mod[:ind_mod])],' '.join(tar[:ind_tar]))
        r_pre += r_scores[0]['rouge-l']['p']
        r_rec += r_scores[0]['rouge-l']['r']
        r_f1 += r_scores[0]['rouge-l']['f']
        i+=increment
        cnt_+=1
    return {'METEOR':meteor_s/float(cnt_),'BLEU': sent_bleu/float(cnt_),'F1': r_f1/float(cnt_), 'Recall': r_rec/float(cnt_), 'Precision': r_pre/float(cnt_)}

def meteor(file_loc, increment=4):
    fp = open(file_loc)
    D = fp.readlines()
    meteor_s = 0.0
    cnt_ = 1e-3
    i=0
    while i<len(D):
        tar = D[i+2].split()[2:]
        mod = D[i+1].split()[1:]

        if '<eos>' in tar:
            ind_tar = tar.index('<eos>')
        else:
            ind_tar = -1
        if '<eos>' in mod:
            ind_mod = mod.index('<eos>')
        else:
            ind_mod = -1
        meteor_s += meteor_score([' '.join(mod[:ind_mod])],' '.join(tar[:ind_tar]))
        i+=increment
        cnt_+=1
    return meteor_s/float(cnt_)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=str)
    parser.add_argument('--increment', type=int, default=4)
    R = Rouge()
    args = parser.parse_args()
    mets = Metrics(args.sample, args.increment)
    print(mets)
    logging.info(mets)
