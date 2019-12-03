import numpy as np
import torch
import torch.nn.functional as F
import os
import threading as thrd
import time
import operator
import torch.nn as nn
from queue import PriorityQueue
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

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.logp < other.logp
        
def beam_decode(decoder, target_tensor, decoder_hiddens, encoder_outputs=None):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = 10
    topk = 3  # how many sentence do you want to generate
    decoded_batch = []

    # decoding goes sentence by sentence
    for idx in range(target_tensor.size(0)):
        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([[SOS_token]], device=device)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            if n.wordid.item() == EOS_token and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch


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
