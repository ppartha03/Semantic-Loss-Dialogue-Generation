import numpy as np
import torch
import torch.nn.functional as F
import os
import threading as thrd
import logging
import re
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#/# Load Embeddings
responses_for_batch = []

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length
        self.decoder_input_d = dec_inp

    def eval(self, alpha=1.0):
        reward = torch.rand(1)
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.logp < other.logp

def getTopK(dec_list, topk = 5, Vocab_inv = None, batch_size = 1,seq_length = 10, thrd_nmb=2):
    global responses_for_batch
    # walk over each step in sequence
    responses_for_batch = [[] for _ in range(batch_size)]
    def batch_decode_thread(thread_ind,data):
        sequences = [[list(), 1.0]]
        count = 0
        print(thread_ind)
        for row in data:
            all_candidates = list()
            for i in range(len(sequences)):
                seq, score = sequences[i]
                if len(seq) == 0 or '<eos>' not in seq[-1]:
                    for j in range(len(row)):
                        candidate = [seq+[Vocab_inv[j]], score * -row[j]]
                        all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup:tup[1])
            # select k best
            sequences = ordered[:topk]
        count+=1
        responses_for_batch[thread_ind] = sequences
    for d in range(len(dec_list)):
        dec_list[d] = dec_list[d].view(batch_size,-1)
    dec_list = torch.stack(dec_list).view(batch_size,seq_length,-1)
    for min_batch in range(0,batch_size,thrd_nmb):
        for bs in range(min_batch,min_batch+thrd_nmb):
            if bs < batch_size:
                thrd.Thread(target = batch_decode_thread, args = [bs, dec_list[bs]]).start()
        for thread in thrd.enumerate():
            if thread.daemon:
                continue
            try:
                thread.join()
            except RuntimeError as err:
                if 'cannot join current thread' in err.args[0]:
                    # catchs main thread
                    continue
                else:
                    raise

    # print(responses_for_batch)
    logging.info(responses_for_batch)
    return responses_for_batch

def Load_embeddings(dataset):
    if dataset == 'mwoz':
        embeddings_file = "mwoz_embedding.txt"
    else:
        embeddings_file = "frames_embedding.txt"
    embs = []
    words = []
    with open(os.path.join("../Dataset", embeddings_file), 'r', encoding="utf8") as f:
        for line in f:
            values = line.split('<del>')
            words.append(values[0])
            coefs = np.asarray(values[1:], dtype='float32')
            embs.append(coefs)
    return words, torch.tensor(embs)


def Bert_loss(output_embeddings, target_embeddings, sentence_embedding):
    if sentence_embedding == "sum":
        output_sen_embedding = torch.sum(output_embeddings, dim=1)
        target_sen_embedding = torch.sum(target_embeddings, dim=1)
    else:
        output_sen_embedding = torch.mean(output_embeddings, dim=1)
        target_sen_embedding = torch.mean(target_embeddings, dim=1)
    mse_loss = F.mse_loss(output_sen_embedding, target_sen_embedding.requires_grad_(False), reduction='mean')
    #mse_loss = torch.mean(mse_loss,dim = 1)
    return mse_loss


def Mask_sentence(res, mask, config):
    # mask_ind here corresponds to the index of the <pad> words
    mask_ind = config['pad_index']
    eos_ind = config['eos_index']
    posteos_mask = config['posteos_mask']
    device = config['device']
    res_masked = res.flatten()
    # mask = torch.from_numpy(mask_).to(device).bool()
    mask = torch.index_select(~mask, dim=0, index=res_masked)
    res_masked = res_masked.masked_fill(mask, mask_ind)
    res_masked = res_masked.reshape_as(res)
    # Apply a mask on the words after the first <eos> generated
    if posteos_mask:
        mask_posteos = res == eos_ind
        indices = np.arange(res.shape[1])
        indices = torch.from_numpy(indices).to(device)
        indices = indices.repeat(res.shape[0], 1)
        eos_positions = indices.masked_fill(~mask_posteos, 100000)
        eos_positions = torch.argmin(eos_positions, dim=1).unsqueeze(1).expand_as(indices)
        mask_posteos = indices <= eos_positions
        res_masked = res_masked.masked_fill(~mask_posteos, mask_ind)
    return res_masked

def Posteos_mask(res, config):
    # mask_ind here corresponds to the index of the <pad> words
    mask_ind = config['pad_index']
    eos_ind = config['eos_index']
    device = config['device']
    res_masked = res
    mask_posteos = res == eos_ind
    indices = np.arange(res.shape[1])
    indices = torch.from_numpy(indices).to(device)
    indices = indices.repeat(res.shape[0], 1)
    eos_positions = indices.masked_fill(~mask_posteos, 100000)
    eos_positions = torch.argmin(eos_positions, dim=1).unsqueeze(1).expand_as(indices)
    mask_posteos = indices <= eos_positions
    res_masked = res_masked.masked_fill(~mask_posteos, mask_ind)
    return res_masked

# Create model id
def create_id(config, saved_models, reload=False, run_id=-1, training_type='train'):
    model_id = '{}'.format(config['dataset'])
    if config['prebert_mask']:
        model_id += '_preBertMask'
    else:
        model_id += '_nopreBertMask'
    model_id += '_{}_{}_{}_{}'.format(config['hidden_size'], config['encoder_learning_rate'],
                                          config['decoder_learning_rate'], config['loss'])
    if config['loss'] == 'combine':
        model_id += '_{}'.format(config['alpha'])
    elif config['loss'] == 'alternate':
        model_id += '_{}'.format(config['toggle'])
    else:
        model_id += '_x'
    model_id += '_{}_{}_{}_run'.format(config['output_dropout'], config['change_nll_mask'], config['sentence_embedding'])

    file_ids = []
    file_id_re = '(?<=' + model_id + ')\d+'

    for filename in os.listdir(saved_models):
        m = re.search(file_id_re, filename)
        if m:
            file_ids.append(int(m.group(0)))

    if training_type == 'train':
        if not reload and not file_ids:
            return model_id + '1', 1
        elif reload and not file_ids:
            raise NameError('No saved models exist')
        elif reload:
            if run_id == -1:
                file_id = max(file_ids)
            else:
                if run_id in file_ids:
                    file_id = run_id
                else:
                    raise NameError('The model you are trying to reload is not there, check the run_id')
        else:
            file_id = max(file_ids) + 1
    else:
        if not file_ids:
            raise NameError('No saved models exist')
        elif run_id == -1:
            file_id = max(file_ids)
        elif run_id in file_ids:
            file_id = run_id
        else:
            raise NameError('The model you are trying to reload is not there, check the run_id')

    model_id += '{}'.format(file_id)
    return model_id, file_id
