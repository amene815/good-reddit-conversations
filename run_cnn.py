from word2vec import load_wv_embedding
import nn_model as nn
import torch
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle, re
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F

class ListDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors) -> None:
        self.tensors = tensors

    def __getitem__(self, index):
        x = tuple(tensor[index] for tensor in self.tensors)
        return x

    def __len__(self):
        return len(self.tensors[0])

def collate(sequence):
    label = []
    feature = []
    for i in range(len(sequence)):
        feature.append(sequence[i][0])
        label.append(sequence[i][1])
    return tuple([feature, label])

def run():
    path = "tokenized-data.csv"
    df = pd.read_csv(path)

    with open('post2words_dict.pickle', 'rb') as handle:
        post2words_dict = pickle.load(handle)

    # load embedding
    embed_dict = load_wv_embedding('embeddedings.txt')

    # load file from embedding and create vector-representataion
    graph_docs = post2words_dict.keys()

    print('Finish Read')

    plt.hist(df['max_len'],bins=70)
    plt.savefig('data.jpg')
    plt.show()
    x = torch.tensor(df['max_len'], dtype=torch.float)
    y = torch.zeros_like(x, dtype=torch.float)

    print(F.mse_loss(x, y))

    # outf = []
    # outl = []
    # for infile in graph_docs:
    #     if infile not in post2words_dict:
    #         continue
    #     try:
    #         word_list = [embed_dict["stoi"][word] for word in post2words_dict[infile]]
    #     except:
    #         raise ValueError("File containing operation out of vocab !!!")
    #     layer_input = torch.tensor(word_list, dtype=torch.long)
    #     file_word_embedding = embed_dict["embedding_layer"](layer_input)
    #     outf.append(file_word_embedding)
    #     outl.append(torch.tensor(df['max_len'][infile]))
    #
    # del graph_docs, post2words_dict, embed_dict
    #
    # X_train, X_val, y_train, y_val = train_test_split(outf, outl, test_size=0.2,
    #                                                   random_state=1)
    #
    # del outl, outf

    # print('Finish splite dataset')
    #
    # trn_dataset = ListDataset(X_train, y_train)
    # trn_data_loader = DataLoader(trn_dataset, 16, shuffle=True, collate_fn=collate)
    #
    # val_dataset = ListDataset(X_train, y_train)
    # val_data_loader = DataLoader(val_dataset, 16, shuffle=False, collate_fn=collate)
    #
    # print('Finish build dataload')
    #
    # nn.main('ffn', trn_data_loader, val_data_loader)

if __name__ == '__main__':
    run()