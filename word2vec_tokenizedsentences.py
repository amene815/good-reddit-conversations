# pylint: disable=no-member
"""
Example run:

    python word2vec.py --save-word2vec-wv js.wv --output-path features_word2vec/js_embeddings.full.csv --epochs 5
"""

import argparse
import os
import time
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import csv


import torch
from gensim.models import Word2Vec

import multiprocessing


parser = argparse.ArgumentParser(description="Run word2Vec.")

parser.add_argument("--output-path",
                    type=str,
                    default='vectorized_data.csv',
                    help="word2vec embeddings path.")

parser.add_argument("--epochs",
                    type=int,
                    default=15,
                    help="Epochs.")

parser.add_argument("--save-word2vec-wv",
                    type=str,
                    default='embeddedings.txt',
                    help="word2vec wv embeddings path.")

parser.add_argument("--dimensions",
                    type=int,
                    default=32,
                    help="Number of dimensions. Default is 32.")

parser.add_argument("--workers",
                    type=int,
                    default=multiprocessing.cpu_count(),
                    help="Number of workers. Default is 4.")

print(parser)
args = parser.parse_args()

def load_wv_embedding(wv_embedding_file: str):
    from gensim.models import KeyedVectors
    import torch

    model = KeyedVectors.load(wv_embedding_file, mmap='r')
    weights = torch.FloatTensor(model.vectors)

    embedding_layer = torch.nn.Embedding.from_pretrained(weights)

    # save stoi itos
    itos = []
    stoi = {}
    #Use KeyedVector's .key_to_index dict, .index_to_key list, and methods .get_vecattr(key, attr) and .set_vecattr(key, attr, new_val) instead.
    # for k in model.key_to_index.keys():
    #     #itos[model.vocab[k]] = k
    #     stoi[k] = model.index_to_key
    itos = model.index_to_key
    stoi = model.key_to_index

    # word count
    w2c = dict()
    for item in stoi.values():
        w2c[item]=model.get_vecattr(item,"count")

    return {"embedding_layer":embedding_layer,
            "itos": itos,
            "stoi": stoi,
            "w2c": w2c}


def main():

    path = "./data/tokenized-data.csv"

    df = pd.read_csv(path)
    re.sub("[\[\]\,\']","",df['stemmed_words'][0]).split(" ")


    post2words_dict = {item[0]:re.sub("[\[\]\,\']","",item[1]['stemmed_words']).split(" ") for item in df.iterrows()}

    corpus = []
    for post in post2words_dict.values():
        corpus.append(post)

    # see detailed parameter settings in https://radimrehurek.com/gensim/models/word2vec.html
    w2v_model = Word2Vec(min_count=1, # Ignores all words with total frequency lower than this
                         window=10, # Maximum distance between the current and predicted word within a sentence
                         vector_size=args.dimensions, # Dimensionality of the word vectors.
                         sample=6e-5,
                         alpha=0.03, # The initial learning rate
                         min_alpha=0.0007, # Learning rate will linearly drop to min_alpha as training progresses
                         negative=20, # negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20)
                         epochs=args.epochs, # Number of iterations (epochs) over the corpus
                         workers=args.workers)

    # build vocab
    t = time.time()
    w2v_model.build_vocab(corpus, progress_per=10000)

    # train model
    t = time.time()
    w2v_model.train(corpus, total_examples=w2v_model.corpus_count, epochs=args.epochs, report_delay=10)

    # save embedding to txt file
    w2v_model.wv.save(args.save_word2vec_wv) #save --> save_word2vec_format

    # load embedding
    embed_dict = load_wv_embedding(args.save_word2vec_wv)

    tokenized_data = [[embed_dict['stoi'][word] for word in sentence] for sentence in post2words_dict.values()]

    X_train, X_val, y_train, y_val = train_test_split(tokenized_data, df['max_len'].tolist(), test_size=0.2, random_state=1)

    with open('data/train.csv', 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter=',')

        for i,x in enumerate(X_train):
            x.append(y_train[i])
            tsv_output.writerow(x)

    with open('data/validation.csv', 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter=',')

        for i,x in enumerate(X_val):
            x.append(y_val[i])
            tsv_output.writerow(x)


if __name__ == "__main__":
    main()
