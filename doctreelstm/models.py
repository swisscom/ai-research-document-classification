"""
Copyright © 2018-present, Swisscom (Schweiz) AG.
All rights reserved.

Author: Khalil Mrini
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

class Model(nn.Module):
    """
    Classifier Parent Module.

    Parameters
    ----------
    embedding_dim: int, embedding dimensions
    hidden_dim: int, hidden dimensions
    label_dim: int, number of labels
    cuda_mode: bool, to activate cuda-based training
    multilabel: bool, True for multi-label classification, False for one label per training element
    """

    def __init__(self, embedding_dim, hidden_dim, label_dim, cuda_mode, multilabel=False):
        super(Model, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.label_dim = label_dim
        self.cuda_mode = cuda_mode and torch.cuda.is_available()
        self.multilabel = multilabel

    def get_prediction(self, output):
        """
        Get prediction given output as Tensor.
        :param output: PyTorch Tensor
        :return: Numpy ndarray if multilabel, else float
        """
        if self.multilabel:
            return np.argwhere(output.data.squeeze().cpu().numpy() >= 0.5)
        else:
            return np.argmax(output.data.squeeze().cpu().numpy())

    def predict(self, samples):
        """
        Predict for multiple samples on a manageable number of datapoints.
        :param samples: list of samples
        :return: predictions
        """
        self.eval()
        labels = []
        for X in tqdm(samples, desc='Predicting...'):
            labels.append(self.get_prediction(self(X)))
        return labels

    def predict_generator(self, generator, length):
        """
        Predict for multiple samples on a generator.
        :param generator: generator of samples
        :param length: number of instances
        :return: predictions
        """
        self.eval()
        labels = []
        for _ in tqdm(length, desc='Predicting...'):
            X = next(generator)
            labels.append(self.get_prediction(self(X)))
        return labels

class StructureTreeLSTM(Model):
    """
    Structure Tree-LSTM Module.

    Parameters
    ----------
    embedding_dim: int, embedding dimensions
    hidden_dim: int, hidden dimensions
    label_dim: int, number of labels
    cuda_mode: bool, to activate cuda-based training
    multilabel: bool, True for multi-label classification, False for one label per training element
    """

    def __init__(self, embedding_dim, hidden_dim, label_dim, cuda_mode, multilabel=False):
        super(StructureTreeLSTM, self).__init__(embedding_dim, hidden_dim, label_dim, cuda_mode, multilabel=multilabel)
        # Matrices:
        # - for the input gate
        self.W_i = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.U_i = nn.Linear(self.hidden_dim, self.hidden_dim)
        # - for the forget gate
        self.W_f = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.U_f = nn.Linear(self.hidden_dim, self.hidden_dim)
        # - for the output gate
        self.W_o = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.U_o = nn.Linear(self.hidden_dim, self.hidden_dim)
        # - for the candidate vector
        self.W_u = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.U_u = nn.Linear(self.hidden_dim, self.hidden_dim)
        # Matrix to transition from hidden layer to labels
        self.hidden2label = nn.Linear(self.hidden_dim, self.label_dim)
        # Showing weights
        self.carry_weights = False

    def unit_forward(self, inputs, children_c, children_h):
        """
        Computing the transition equations for the current unit.
        :param inputs: torch variable, vector inputs of the unit
        :param children_c: torch variable, memory cells of the children units
        :param children_h: torch variable, hidden states of the children units
        :return c: torch variable, memory cell
        :return h: torch variable, hidden state
        """
        children_h_sum = torch.sum(children_h, dim=0, keepdim=True)
        i = F.sigmoid(self.W_i(inputs) + self.U_i(children_h_sum))
        f = F.sigmoid(self.W_f(inputs).repeat(len(children_h), 1) + self.U_f(children_h))
        o = F.sigmoid(self.W_o(inputs) + self.U_o(children_h_sum))
        u = F.tanh(self.W_u(inputs) + self.U_u(children_h_sum))
        carried_from_children = torch.mul(f, children_c)
        c = torch.mul(i, u) + torch.sum(carried_from_children, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        if self.carry_weights:
            return [c, h, carried_from_children]
        else:
            return c, h

    def forward(self, tree):
        """
        Forwarding for the whole document tree.
        :param tree: DocTreeNode, the document tree
        :return: torch variable, hidden state
        """
        if self.carry_weights:
            children_carries = []
        for idx in range(tree.num_children):
            if self.carry_weights:
                _, child_carry = self.forward(tree.children[idx])
                children_carries.append(children_carries)
            else:
                self.forward(tree.children[idx])

        if tree.embedding is not None:
            variable = Variable(torch.Tensor(tree.embedding))
        else:
            variable = Variable(torch.Tensor(np.zeros(self.embedding_dim)))
        if self.cuda_mode:
            variable = variable.cuda()

        if tree.num_children == 0:
            # Fill with a tensor (vector) of just zeroes
            child_c = Variable(variable.data.new(1, self.hidden_dim).fill_(0.))
            child_h = Variable(variable.data.new(1, self.hidden_dim).fill_(0.))
        else:
            # Here, children have taken on a "state"
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        if self.cuda_mode:
            child_c = child_c.cuda()
            child_h = child_h.cuda()

        if self.carry_weights:
            c, h, carried_from_children = self.unit_forward(variable, child_c, child_h)
            if tree.num_children == 0:
                carried_from_children = None
            carried = [carried_from_children] + children_carries
            tree.state = c, h
        else:
            tree.state = self.unit_forward(variable, child_c, child_h)
        doc_output = self.hidden2label(tree.state[1])[0]

        if self.multilabel:
            doc_output = F.sigmoid(doc_output)

        if self.carry_weights:
            return doc_output, carried
        else:
            return doc_output

    def get_weight_list(self, weights):
        """
        Get attention weights, recursive function.
        :param weights: PyTorch Tensor, or list of such instance, or None
        :return: at the end of the recursion, each node is represented by a list of weights of its children,
                 which sum is 1
        """
        if weights is None:
            return -1
        elif type(weights) == list:
            return [self.get_weight_list(child) for child in weights]
        else:
            weights = weights.data.squeeze().cpu().numpy()
            weights = np.divide(weights, np.sum(weights, axis=-1))
            weights = np.mean(weights, axis=0)
            return weights

    def add_weights_to_tree(self, doctree, weights):
        """
        Adds weights to doctree recursively.
        :param doctree: DocTreeNode instance
        :param weights: list of weights
        :return: None
        """
        doctree.children_weights = weights[0]
        for child_idx in range(doctree.num_children):
            self.add_weights_to_tree(doctree.children[child_idx], weights[child_idx+1])

    def predict_with_weights(self, doctree, get_weight_list=False, get_weight_tree=False):
        """
        Predict for one sample, and optionally get weights.
        :param doctree: DocTreeNode, document tree
        :param get_weight_list: bool, whether to carry weights and return them (True) or not (False)
        :param get_weight_tree: bool, whether to add weights to the doctree and return it
        :return: prediction
        """
        if get_weight_list or get_weight_tree:
            self.carry_weights = True
            prediction, weights = self(doctree)
            self.carry_weights = False
            prediction = self.get_prediction(prediction)
            weights = self.get_weight_list(weights)
            returned = [prediction]
            if get_weight_list:
                returned.append(weights)
            if get_weight_tree:
                self.add_weights_to_tree(doctree, weights)
                returned.append(doctree)
            return returned
        else:
            return self.get_prediction(self(doctree))

class LSTMClassifier(Model):
    """
    Classifier with an LSTM layer.
    The input should be a document represented by a sequence of vectors.

    Parameters
    ----------
    embedding_dim: int, embedding dimensions
    hidden_dim: int, hidden dimensions
    label_dim: int, number of labels
    cuda_mode: bool, to activate cuda-based training
    multilabel: bool, True for multi-label classification, False for one label per training element
    bidirectional: bool, for a bi-LSTM
    lstm_layers: int, number of layers in the LSTM
    """

    def __init__(self, embedding_dim, hidden_dim, label_dim, cuda_mode, multilabel=False, bidirectional=False, lstm_layers=1):
        super(LSTMClassifier, self).__init__(embedding_dim, hidden_dim, label_dim, cuda_mode, multilabel=multilabel)
        self.hidden_dim = (int(bidirectional) + 1) * hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden2label = nn.Linear(hidden_dim, label_dim)

    def init_hidden(self):
        """
        Initiates the hidden variable and memory cell.
        :return: the two initialised variables as Torch Variables
        """
        h0 = Variable(torch.zeros(self.lstm_layers, 1, self.hidden_dim))
        c0 = Variable(torch.zeros(self.lstm_layers, 1, self.hidden_dim))
        if self.cuda_mode:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)

    def forward(self, sequential_doc):
        """
        Forwarding for the sequential document.
        :param sequential_doc: list of embeddings
        :return: torch variable, hidden state
        """
        doc_input = Variable(torch.FloatTensor(np.array([sequential_doc])))
        if self.cuda_mode:
            doc_input = doc_input.cuda()
        lstm_out, self.hidden = self.lstm(doc_input, self.init_hidden())
        doc_output = self.hidden2label(lstm_out[0,-1])
        if self.multilabel:
            doc_output = F.sigmoid(doc_output)
        return doc_output

class MLPClassifier(Model):
    """
    Classifier with two linear layers.
    The input should be a document represented by one vector.

    Parameters
    ----------
    embedding_dim: int, embedding dimensions
    hidden_dim: int, hidden dimensions
    label_dim: int, number of labels
    cuda_mode: bool, to activate cuda-based training
    multilabel: bool, True for multi-label classification, False for one label per training element
    """

    def __init__(self, embedding_dim, hidden_dim, label_dim, cuda_mode, multilabel=False):
        super(MLPClassifier, self).__init__(embedding_dim, hidden_dim, label_dim, cuda_mode, multilabel=multilabel)
        self.layer1 = nn.Linear(embedding_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, label_dim)

    def forward(self, doc_embedding):
        """
        Forwarding for the document embedding.
        :param doc_embedding: list of real numbers (embedding)
        :return: torch variable, hidden state
        """
        doc_input = Variable(torch.FloatTensor(doc_embedding))
        if self.cuda_mode:
            doc_input = doc_input.cuda()
        doc_output = self.layer1(doc_input)
        doc_output = self.act1(doc_output)
        doc_output = self.layer2(doc_output)
        if self.multilabel:
            doc_output = F.sigmoid(doc_output)
        return doc_output