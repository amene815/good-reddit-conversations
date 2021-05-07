import torch
from torch import nn

import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt

from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import time
import sys

from csv import reader

from sklearn.feature_extraction.text import CountVectorizer

class RNN(nn.Module):
    def __init__(self,vocab_size, embed_size,input_size, hidden_size, drop_rate, num_layers):
        super(RNN, self).__init__()
        # ---------------------------------
        # Configuration
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(drop_rate)
        self.num_layers = num_layers
        self.input_dim = input_size
        self.hidden_dim = hidden_size
        # ---------------------------------
        # Model Arch
        self.lstm = nn.RNN(embed_size, hidden_size,dropout=drop_rate, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, input, hidden):
        # Input is a list with element Dim: L_i x E
        # length = torch.tensor([len(i) for i in input])

        # ---------------------------------
        # === Hidden layer ===
        # padded = pad_sequence(input)
        # x = pack_padded_sequence(padded,length, enforce_sorted=False) # Dim: T x B x E

        input = self.embed(input)
        out, hidden_out = self.lstm(input,hidden)

        # ---------------------------------
        # === Classification layer ===
        fc_out = self.fc(out)  # Dim: Batch_size x 1
        return fc_out, hidden_out
    
    def init_hidden(self, batch_size):

        if (torch.cuda.is_available()):
          h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)).cuda() #hidden state
          c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)).cuda() #internal state
        else:
          h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)) #hidden state
          c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)) #internal state


        return (h_0,c_0)



def eval(data_iter, model, criterion):
    """ Evaluate the model with the data
    data_iter: the evaluate data
    model: the defined model
    """
    # set in the eval model, which will trun off the features only used for training, such as droput
    model.eval()
    # records
    val_loss, val_batch = 0, 0
    total_example, correct_pred = 0, 0
    # iterate all the mini batches for evaluation
    for b, batch in enumerate(data_iter):

        input = batch[:,:-1]
        target = batch[:,-1]
        (h,c) = model.init_hidden(batch_size)

        if(batch.size()[0] != h.shape[1]):
            (h,c) = model.init_hidden(batch.size()[0])
            input = batch[:,:32].view(batch_size,1,32)
            target = batch[:,-1]
        # initialize optimizer
        # forward: prediction
        output, (h,c) = model(input, (h,c))
        # loss function
        output = torch.flatten(output,start_dim=0,end_dim=1)
        loss = criterion(output, target)

    return val_loss / val_batch


def main(device):
    """
    Both train_iter and val_iter can be is list; each element in the list is (input, label).
    For FFN,  the 'input' is the vector of a batch of docs (e.g., take the mean of all word embedding) B x E.
    For both CNN and RNN, the 'input' is a list of different-sequence-length word embedding of each doc (e.g., [L_1 x E, L_2 x E, ..., L_B x E).
    B is the batch size, E is the word embedding size.
    """
    # if model_name == 'ffn':
    #     model = NeuralClassifier(embed_size=32, feature_size=16, drop_rate=0.0)
    # elif model_name == 'cnn':
    #     model = CNN(embed_size=32, feature_size=16, drop_rate=0.0)
    # elif model_name == 'rnn':
    #     model = RNN(embed_size=32, feature_size=16, drop_rate=0.0)
    # else:
    #     raise ValueError("Unrecognized model name")


    #Data Processing
    # df = pd.read_csv("data/train.csv", header=None)

    # post_embeddings = df.to_numpy()

    with open('data/train.csv','r') as read_obj:
        csv_reader = reader(read_obj)
        sentences = []
        for row in csv_reader:
            sentences.append([int(x) for x in row])

    with open('data/validation.csv','r') as read_obj:
        csv_reader = reader(read_obj)
        val = []
        for row in csv_reader:
            val.append([int(x) for x in row])

    # Data processing
    # post_embeddings = [np.asarray(re.sub("[\[\]\,\']","",x).split(' '),dtype=np.float64) for x in post_embeddings]
    
    
    def collate(batch):
        # batch = [torch.LongTensor(x) for x in batch]
        input_batch = [x[:-1] for x in batch]
        target_batch = [x[-1] for x in batch]
        input_batch = [torch.LongTensor(x) for x in input_batch]
        target_batch = torch.LongTensor(target_batch)
        batch_padded = pad_sequence(input_batch,batch_first=True,padding_value=pad_token)
        batch_final = torch.cat((batch_padded,target_batch.view(8,1)),1)
        return batch_final.to(device)
    
    batch_size = 8
    vocab_size = 124581
    pad_token = 124580

    train_data = torch.utils.data.DataLoader(sentences, batch_size=batch_size, shuffle=True, collate_fn=collate)

    train_iter = iter(train_data)

    val_data = torch.utils.data.DataLoader(sentences, batch_size=batch_size, shuffle=True, collate_fn=collate)

    val_iter = iter(val_data)


    model = RNN(vocab_size = vocab_size, embed_size = 32,input_size=32,hidden_size=128, drop_rate=0.3,num_layers=1)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.MSELoss()

    epoch, val_step = 5, 50

    TrnLoss, ValLoss, ValAcc = [], [], []
    start_time, stop_time, time_diff, total_time = 0,0,0,0
    total_batch = 0
    torch.autograd.set_detect_anomaly(True)
    for e in range(epoch):
        # print(e)
        for b, batch in enumerate(train_iter):
            start_time = time.perf_counter()
            total_batch += 1
            input = batch[:,:-1]
            target = batch[:,-1]
            (h,c) = model.init_hidden(batch_size)

            if(batch.size()[0] != h.shape[1]):
                (h,c) = model.init_hidden(batch.size()[0])
                input = batch[:,:32].view(batch_size,1,32)
                target = batch[:,-1]

            
            # Update parameters with one batch
            # set in training mode
            model.train()
            # initialize optimizer
            optimizer.zero_grad()
            # forward: prediction
            output, h = model(input, h)
            # loss function
            output = torch.flatten(output,start_dim=0,end_dim=1)
            loss = criterion(output, target)
            print("Loss: {}".format(loss))
            # backward: gradient computation
            loss.backward()
            h.detach()
            c.detach()
            # norm clipping, in case the gradient norm is too large
            clip_grad_norm(model.parameters(), 1.0)
            # gradient-based update parameter
            optimizer.step()
            # Compute validation loss after each val_step
            if total_batch % val_step == 0:
                val_loss = eval(val_iter, model, criterion)
                ValLoss.append(val_loss)
                TrnLoss.append(loss)

                toolbar_width = 50
                # setting up toolbar [-------------------------------------]
                sys.stdout.write("[%s]"%(("-")*toolbar_width))
                sys.stdout.flush()
                # each hash represents 2 % of the progress
                sys.stdout.write("\r") # return to start of line
                sys.stdout.flush()
                sys.stdout.write("Epoch: {} out of {}, Batch: {} out of {}, Estimated Time: {}  [".format(e,epoch,b,len(train_iter),time_est))#Overwrite over the existing text from the start 
                sys.stdout.write("#"*(current_progress+1))# number of # denotes the progress completed 
                sys.stdout.flush()
            
            stop_time = time.perf_counter()
            time_diff = stop_time-start_time
            total_time += time_diff

            running_avg = total_time/total_batch

            time_est = running_avg*(len(train_iter)-total_batch)
            current_progress = int(round((total_batch/len(train_iter))*50))
            



    print("The best validation accuracy = {:.4}".format(max(ValAcc)))

    plt.plot(range(len(TrnLoss)), TrnLoss, color="red", label="Training Loss")  # Training loss
    plt.plot(range(len(ValLoss)), ValLoss, color="blue", label="Development Loss")  # Val loss
    plt.xlabel("Steps")
    plt.ylabel("MSE")
    plt.legend()

    torch.save(model,'reddit-language-model.pth')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
main(device)