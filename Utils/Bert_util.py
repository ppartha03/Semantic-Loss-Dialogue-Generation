import numpy as np
import torch
import torch.nn.functional as F
import os
import threading as thrd
import time

#/# Load Embeddings
responses_for_batch = []
def getTopK(dec_list, topk = 5, Vocab_inv = None, batch_size = 1,seq_length = 10):
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
    for min_batch in range(0,batch_size,2):
        for bs in range(min_batch,min_batch+2):
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

    print(responses_for_batch)
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


def Bert_loss(output_embeddings, target_embeddings):
    output_sen_embedding = torch.mean(output_embeddings, dim=1)
    target_sen_embedding = torch.mean(target_embeddings, dim=1)
    mse_loss = F.mse_loss(output_sen_embedding, target_sen_embedding.requires_grad_(False), reduction='mean')
    #mse_loss = torch.mean(mse_loss,dim = 1)
    return mse_loss


def Mask_sentence(res, mask_, mask_ind=3, device=None):
    res_masked = res.flatten()
    mask = torch.from_numpy(mask_).to(device).bool()
    mask = torch.index_select(~mask, dim=0, index=res_masked)
    res_masked = res_masked.masked_fill(mask, mask_ind)
    res_masked = res_masked.reshape_as(res)
    return res_masked
