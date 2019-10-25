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
    mse_loss = F.mse_loss(output_sen_embedding, target_sen_embedding.requires_grad_(False), reduction='mean')
    #mse_loss = torch.mean(mse_loss,dim = 1)
    return mse_loss


def Mask_sentence(res, mask_, config):
    mask_ind = config['pad_index']
    eos_ind = config['eos_index']
    no_posteos_mask = config['no_posteos_mask']
    device = config['device']
    res_masked = res.flatten()
    mask = torch.from_numpy(mask_).to(device).bool()
    mask = torch.index_select(~mask, dim=0, index=res_masked)
    res_masked = res_masked.masked_fill(mask, mask_ind)
    res_masked = res_masked.reshape_as(res)
    # Apply a mask on the words after the first <eos> generated
    if not no_posteos_mask:
        posteos_mask = res == eos_ind
        indices = np.arange(res.shape[1])
        indices = torch.from_numpy(indices)
        indices = indices.repeat(res.shape[0], 1)
        eos_positions = indices.masked_fill(~posteos_mask, 100000)
        eos_positions = torch.argmin(eos_positions, dim=1).unsqueeze(1).expand_as(indices)
        posteos_mask = indices <= eos_positions
        res_masked = res_masked.masked_fill(~posteos_mask, mask_ind)
    return res_masked
