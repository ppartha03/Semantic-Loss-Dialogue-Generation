import numpy as np
import torch
import torch.nn.functional as F
import os

#/# Load Embeddings
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
    mse_loss = F.mse_loss(output_sen_embedding, target_sen_embedding, reduction='none')
    mse_loss = torch.mean(mse_loss,dim = 1)
    return mse_loss

def Mask_sentence(res, tar, mask_, mask_ind=3):
    res_masked = res.flatten()
    tar_masked = tar.flatten()
    mask = ~torch.Tensor(mask_).bool()
    mask = torch.index_select(mask, dim=0, index=tar_masked)
    tar_masked.masked_fill_(mask, mask_ind)
    res_masked.masked_fill_(mask, mask_ind)
    tar_masked = tar_masked.reshape_as(tar)
    res_masked = res_masked.reshape_as(res)
    return res_masked, tar_masked
