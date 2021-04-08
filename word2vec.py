# pylint: disable=no-member
"""
Example run:

    python word2vec.py --tmp-dir features_word2vec \
        --save-word2vec-wv js.wv \
        --input-dataset datasets/prod/js/embeddings_full.json \
        --output-path features_word2vec/js_embeddings.full.csv \
        --save-model-path features_word2vec/js_embeddings.full.model
"""

import argparse
import os
import time
import pandas as pd

import torch
from gensim.models import Word2Vec

import multiprocessing


parser = argparse.ArgumentParser(description="Run word2Vec.")

parser.add_argument("--input-paths",
                    nargs="+",
                    help="Input folders with jsons.")

parser.add_argument("--input-dataset",
                    type=str,
                    help="Input dataset spec")

parser.add_argument("--output-path",
                    type=str,
                    required=True,
                    help="word2vec embeddings path.")

parser.add_argument("--epochs",
                    type=int,
                    required=True,
                    help="Epochs.")

parser.add_argument("--save-word2vec-wv",
                    type=str,
                    required=True,
                    help="word2vec wv embeddings path.")

parser.add_argument("--save-model-path",
                    type=str,
                    help="Save model path.")

parser.add_argument("--tmp-dir",
                    type=str,
                    default="features/",
                    help="temporary directory for wv embedding file")

parser.add_argument("--dimensions",
                    type=int,
                    default=32,
                    help="Number of dimensions. Default is 128.")

parser.add_argument("--workers",
                    type=int,
                    default=multiprocessing.cpu_count(),
                    help="Number of workers. Default is 4.")


args = parser.parse_args()

def load_wv_embedding(wv_embedding_file: str):
    from gensim.models import KeyedVectors
    import torch

    model = KeyedVectors.load(wv_embedding_file, mmap='r')
    weights = torch.FloatTensor(model.vectors)

    embedding_layer = torch.nn.Embedding.from_pretrained(weights)

    # save stoi itos
    itos = {}
    stoi = {}
    for k in model.wv.vocab.keys():
        itos[model.wv.vocab[k].index] = k
        stoi[k] = model.wv.vocab[k].index

    # word count
    w2c = dict()
    for item in model.wv.vocab:
        w2c[item]=model.wv.vocab[item].count

    return {"embedding_layer":embedding_layer,
            "itos": itos,
            "stoi": stoi,
            "w2c": w2c}

def main():

    # TODO: Connect with preprocessed data, write function to building corpus
    post2words_dict, corpus = build_post2word_dict(args)

    # see detailed parameter settings in https://radimrehurek.com/gensim/models/word2vec.html
    w2v_model = Word2Vec(min_count=1, # Ignores all words with total frequency lower than this
                         window=10, # Maximum distance between the current and predicted word within a sentence
                         size=args.dimensions, # Dimensionality of the word vectors.
                         sample=6e-5,
                         alpha=0.03, # The initial learning rate
                         min_alpha=0.0007, # Learning rate will linearly drop to min_alpha as training progresses
                         negative=20, # negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20)
                         iter=args.epochs, # Number of iterations (epochs) over the corpus
                         workers=args.workers)

    # build vocal
    t = time.time()
    w2v_model.build_vocab(corpus, progress_per=10000)

    # train model
    t = time.time()
    w2v_model.train(corpus, total_examples=w2v_model.corpus_count, epochs=args.epochs, report_delay=10)

    # save embedding to txt file
    w2v_model.wv.save(args.save_word2vec_wv) #save --> save_word2vec_format

    # load embedding
    embed_dict = load_wv_embedding(args.save_word2vec_wv)

    # load file from embedding and create vector-representataion
    out = []
    for infile in graph_docs:
        if infile not in post2words_dict:
            continue
        try:
            word_list = [embed_dict["stoi"][word] for word in post2words_dict[infile]]
        except:
            raise ValueError("File containing operation out of vocal !!!")
        layer_input = torch.LongTensor(word_list)
        file_word_embedding = embed_dict["embedding_layer"](layer_input)
        file_embedding = torch.mean(file_word_embedding, dim=0).tolist() # average the embedding for each word
        out.append([infile] + file_embedding)

    # save to csv file
    column_names = ["filepath"] + ["x_" + str(dim) for dim in range(args.dimensions)]
    out = pd.DataFrame(out, columns=column_names)
    out.fillna(0, inplace=True) # FIXME: this is a HACK?
    out = out.sort_values(["filepath"])
    out.to_csv(args.output_path, index=None)

if __name__ == "__main__":
    main()
