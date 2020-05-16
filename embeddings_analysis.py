from Model.RNN import EncoderRNN, AttnDecoderRNN
from Utils.Bert_util import Load_embeddings
from Dataset_utils.Frames_data_iterator import FramesGraphDataset
from Dataset_utils.WoZ_data_iterator import WoZGraphDataset
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import logging
import csv
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="frames", choices=["frames", "mwoz"])
parser.add_argument('--results_root', type=str, default='.')
parser.add_argument('--data_path', type=str, default="./Dataset")
# which model to use for validation/test, 'best_mle' or 'best_combined',
# 'best_meteor'
parser.add_argument('--validation_model', type=str, default='best_bleu')#,
                    #choices=["best_mle", "best_combined_loss", "best_meteor", "best_bleu", "last"])
parser.add_argument('--n_words', type=int, default=50)
parser.add_argument('--n_nearest_words', type=int, default=10)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plotHexbin(LM,epoch,filename, X_r):
    fig, axs = plt.subplots(ncols=1, sharey=True, figsize=(7, 4))
    fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)

    hb = axs.hexbin(X_r[:,0], X_r[:,1], gridsize=80, bins='log', cmap='YlOrBr')
    ax.axis([-8,10.5,-7,10])
    axs.set_title(LM +' word embeddings distribution after '+epoch+' epochs')
    cb = fig.colorbar(hb, ax=axs)
    cb.set_label('counts')

class Seq2Seq(nn.Module):
    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        # Encoder_model can be s2s or hred
        self.config = config
        self.mask = np.hstack(
            [config['mask_special_indices'],
             np.ones(config['input_size'] - len(config['mask_special_indices']))])
        self.Encoder = EncoderRNN(self.config['input_size'], self.config['hidden_size'], self.config['num_layers']).to(
            self.config['device'])
        self.Decoder = AttnDecoderRNN(self.config['hidden_size'], self.config['output_size'], self.config['num_layers'],
                                      self.config['sequence_length']).to(self.config['device'])
        _, self.weights = Load_embeddings(config['dataset'], config['embeddings'], config['data_path'])
        self.Sem_embedding = nn.Embedding.from_pretrained(
            self.weights, freeze=True).to(self.config['device'])
        # Loss and optimizer
        self.criterion = nn.NLLLoss(
            weight=torch.from_numpy(
                self.mask).float()).to(self.config['device'])
        # criterion_2 = nn.CrossEntropyLoss().to(self.config['device'])

        self.optimizer = torch.optim.RMSprop(
            self.Encoder.parameters(),
            lr=config['encoder_learning_rate'],
            alpha=0.99,
            eps=1e-08,
            weight_decay=0,
            momentum=0,
            centered=False)
        self.optimizer_dec = torch.optim.Adam(
            self.Decoder.parameters(),
            lr=config['decoder_learning_rate'])
        self.Opts = [self.optimizer, self.optimizer_dec]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    if args.dataset == 'mwoz':
        Data_train = WoZGraphDataset(
            Data_dir=args.data_path + '/MULTIWOZ2/')
    else:
        Data_train = FramesGraphDataset(
            Data_dir=args.data_path + '/Frames-dataset/')

    mapdict = {20: 'Baseline', 23: 'BERT', 30: 'fastText', 31: 'GloVe'}
    seeds = [101, 102, 103]

    # Sample words
    np.random.seed(100)
    sampled_words_indices = np.random.choice(len(Data_train.Vocab), args.n_words, replace=False)
    sampled_words = [Data_train.Vocab_inv[ind] for ind in sampled_words_indices]
    sampled_words_dict = {word:{v:{} for _, v in mapdict.items()} for word in sampled_words}
    n_nearest_words = args.n_nearest_words
    pca = PCA(n_components=2)
    # Get nearest words
    for k, v in mapdict.items():
        for seed in seeds:
            run_id = "exp_" + str(k) + "_seed_" + str(seed)
            run_path = os.path.join(args.results_root, "Results", args.dataset, run_id)

            saved_models = os.path.join(run_path, 'Saved_Models')

            try:
                checkpoint = torch.load(
                    os.path.join(saved_models, run_id + '_' + args.validation_model))
                logging.info("Loading model " + run_id + '_' + args.validation_model)
                for word in sampled_words:
                    sampled_words_dict[word][v][seed] = []
            except:
                logging.info("Couldn't load model " + run_id + '_' + args.validation_model)
                continue

            config = checkpoint['config']
            Model = Seq2Seq(config)
            Model.load_state_dict(checkpoint['model_State_dict'])
            # PCA with sklearn
            words_embeddings = Model.Decoder.embedding.weight.data.to(device)
            X_r = pca.fit(words_embeddings.cpu()).transform(words_embeddings.cpu())
            plotHexbin(v,args.validation_model,os.path.join('./PCA_plots',args.dataset+'_'+args.validation_model+'_exp_'+v+'_seed_'+str(seed)+'.png',X_r)
            sampled_words_embeddings = words_embeddings[sampled_words_indices, :]
            cos_similarities = nn.functional.cosine_similarity(
                sampled_words_embeddings.unsqueeze(1).expand(-1, words_embeddings.shape[0], -1),
                words_embeddings.unsqueeze(0).expand(args.n_words, -1, -1), dim=-1)

            nearest_words_distances, nearest_words_indices = \
                cos_similarities.topk(n_nearest_words + 1, dim=-1, largest=True)
            nearest_words_distances = nearest_words_distances[:, 1:]
            nearest_words_indices = nearest_words_indices[:, 1:]
            for i in range(sampled_words_embeddings.shape[0]):
                word = sampled_words[i]
                for nearest_word_ind, nearest_word_dist in zip(nearest_words_indices[i], nearest_words_distances[i]):
                    nearest_word = Data_train.Vocab_inv[nearest_word_ind.item()]
                    sampled_words_dict[word][v][seed].append((nearest_word, round(nearest_word_dist.item(), 2)))

    # Write the sampled_words_dict in a csv file
    fieldnames = ['word', 'experiment', 'seed'] + [str(d) + ' nearest word' for d in range(1, args.n_nearest_words + 1)]
    target = open("embeddings_analysis_"+args.validation_model+".csv", "w")
    writer = csv.DictWriter(target, fieldnames=fieldnames)
    writer.writerow(dict(zip(fieldnames, fieldnames)))
    for word, experiments in sampled_words_dict.items():
        for exp, seeds in experiments.items():
            for seed, nearest_words in seeds.items():
                writer.writerow(dict(zip(fieldnames, [word, exp, seed] + nearest_words)))
