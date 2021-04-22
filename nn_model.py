import torch
from torch import nn

from matplotlib import pyplot as plt

from torch import optim
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class NeuralClassifier(nn.Module):
    def __init__(self, embed_size, feature_size, drop_rate=0.0):
        super(NeuralClassifier, self).__init__()
        """ Initialization
        - vocab_size
        - embed_size: word embedding size
        - drop_rate: dropout rate
        - class_size: number of classes. For binary classification, class_size = 2
        """
        # ---------------------------------
        # Configuration
        self.dropout = nn.Dropout(drop_rate)

        # ---------------------------------
        # Network parameters
        self.fc1 = nn.Linear(embed_size, feature_size, bias=True)
        self.fc2 = nn.Linear(feature_size, feature_size//2, bias=True)
        self.fc3 = nn.Linear(feature_size//2, 1, bias=True)

    def forward(self, input, label):
        """ Forward function
        """
        x = self.fc1(input)  # Dim:  B x E1
        x = self.dropout(x)
        hidden = F.relu(x)  # Dim: B x E1

        hidden = self.fc2(hidden)  # Dim:  B x E2
        hidden = self.dropout(hidden)
        hidden = F.relu(hidden)  # Dim: B x E2

        logit = self.fc3(hidden)  # Dim:  B x 1

        # ---------------------------------
        # === Loss function ===
        loss = F.mse_loss(logit.squeeze(), label.squeeze())
        return loss, logit


class CNN(nn.Module):
    def __init__(self, embed_size, feature_size, drop_rate=0.0, kernel_sizes=None):
        super(CNN, self).__init__()
        # ---------------------------------
        # Configuration
        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4]
        self.embed_size = embed_size
        self.feature_size = feature_size
        self.kernel_sizes = kernel_sizes  # a list of kernel sizes
        self.dropout = nn.Dropout(drop_rate)
        # ---------------------------------
        # Model Arch
        # Remove the following line and define some parameters for the CNN model here
        self.cov1 = torch.nn.Conv1d(embed_size, feature_size, kernel_sizes[0])
        self.cov2 = torch.nn.Conv1d(embed_size, feature_size, kernel_sizes[1])
        self.cov3 = torch.nn.Conv1d(embed_size, feature_size, kernel_sizes[2])

        self.fc = nn.Linear(feature_size * 3, 1, bias=True)

    def forward(self, input, label):
        # Input is a list with element Dim: L_i x E

        # ---------------------------------
        # === Hidden layer ===
        x = pad_sequence(input).permute(1, 2, 0) # B x E x L

        hidden1 = F.relu(self.cov1(x)).squeeze()
        hidden2 = F.relu(self.cov2(x)).squeeze()
        hidden3 = F.relu(self.cov3(x)).squeeze()

        pooling1, _ = torch.max(hidden1, dim=-1)
        pooling2, _ = torch.max(hidden2, dim=-1)
        pooling3, _ = torch.max(hidden3, dim=-1)

        hidden = torch.cat((pooling1, pooling2, pooling3), 1)

        # ---------------------------------
        # === Classification layer ===
        logit = self.fc(hidden)  # Dim: Batch_size x 1

        loss = F.mse_loss(logit.squeeze(), label.squeeze())
        return loss, logit


class RNN(nn.Module):
    def __init__(self, embed_size, feature_size, drop_rate=0.0):
        super(RNN, self).__init__()
        # ---------------------------------
        # Configuration
        self.embed_size = embed_size
        self.feature_size = feature_size
        self.dropout = nn.Dropout(drop_rate)
        # ---------------------------------
        # Model Arch
        self.rnn1 = nn.RNN(embed_size, feature_size, nonlinearity='relu',dropout=drop_rate)
        self.rnn2 = nn.RNN(feature_size, feature_size//2, nonlinearity='relu', dropout=drop_rate)
        self.fc = nn.Linear(feature_size//2, 1, bias=True)

    def forward(self, input, label):
        # Input is a list with element Dim: L_i x E
        length = torch.tensor([len(i) for i in input])

        # ---------------------------------
        # === Hidden layer ===
        padded = pad_sequence(input)
        x = pack_padded_sequence(padded,length, enforce_sorted=False) # Dim: T x B x E

        hidden, h_n = self.rnn1(x)
        hidden, h_n = self.rnn2(hidden)
        h_n = h_n[-1]

        # ---------------------------------
        # === Classification layer ===
        logit = self.fc(h_n)  # Dim: Batch_size x 1

        loss = F.mse_loss(logit.squeeze(), label.squeeze())
        return loss, logit


def batch_train(input, label, model, optimizer):
    """ Training with one batch
    - batch: a min-batch of the data
    - model: the defined neural network
    - optimizer: optimization method used to update the parameters
    """
    # set in training mode
    model.train()
    # initialize optimizer
    optimizer.zero_grad()
    # forward: prediction
    loss, _ = model(input, label)
    # backward: gradient computation
    loss.backward()
    # norm clipping, in case the gradient norm is too large
    clip_grad_norm(model.parameters(), 1.0)
    # gradient-based update parameter
    optimizer.step()
    return model, loss.item()


def eval(data_iter, model):
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
        # Forward: prediction
        loss, logprob = model(batch[0], batch[1])

        val_batch += 1
        val_loss += loss

    return val_loss / val_batch


def main(model_name):
    """
    Both train_iter and val_iter can be is list; each element in the list is (input, label).
    For FFN,  the 'input' is the vector of a batch of docs (e.g., take the mean of all word embedding) B x E.
    For both CNN and RNN, the 'input' is a list of different-sequence-length word embedding of each doc (e.g., [L_1 x E, L_2 x E, ..., L_B x E).
    B is the batch size, E is the word embedding size.
    """
    if model_name == 'ffn':
        model = NeuralClassifier(embed_size=32, feature_size=16, drop_rate=0.0)
    elif model_name == 'cnn':
        model = CNN(embed_size=32, feature_size=16, drop_rate=0.0)
    elif model_name == 'rnn':
        model = RNN(embed_size=32, feature_size=16, drop_rate=0.0)
    else:
        raise ValueError("Unrecognized model name")

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

    epoch, val_step = 5, 50

    TrnLoss, ValLoss, ValAcc = [], [], []
    total_batch = 0
    for e in trange(epoch):
        # print(e)
        for b, batch in enumerate(train_iter):
            total_batch += 1
            # Update parameters with one batch
            model, loss = batch_train(batch[0], batch[1], model, optimizer)
            # Compute validation loss after each val_step
            if total_batch % val_step == 0:
                val_loss = eval(val_iter, model)
                ValLoss.append(val_loss)
                TrnLoss.append(loss)
    print("The best validation accuracy = {:.4}".format(max(ValAcc)))

    plt.plot(range(len(TrnLoss)), TrnLoss, color="red", label="Training Loss")  # Training loss
    plt.plot(range(len(ValLoss)), ValLoss, color="blue", label="Develoopment Loss")  # Val loss
    plt.xlabel("Steps")
    plt.ylabel("MSE")
    plt.legend()

