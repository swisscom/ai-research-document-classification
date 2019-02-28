"""
Copyright © 2018-present, Swisscom (Schweiz) AG.
All rights reserved.

Author: Khalil Mrini
"""

from tqdm import tqdm
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch.optim as optim

from gensim.models import KeyedVectors
from itertools import cycle

from .utils import form_doctree_by_dfs, form_embedding_sequence, accuracy, micro_f1, macro_f1, all_f1s
from .losses import CustomCrossEntropyLoss, CustomBCELoss
from .doctree import DocTreeNode
from .models import StructureTreeLSTM, LSTMClassifier, MLPClassifier

import logging

# Model types
TREE_MODEL = 1
LSTM_MODEL = 2
MLP_MODEL = 3

class Trainer(object):
    """
    Training object class.
    It takes as parameters a model to train.

    Parameters
    ----------
    model: from the Model class, to train
    batch_size: int, training batch size
    optimizer: pytorch optimizer to use
    learning_rate: float
    weight_decay: float
    target_replication: None if no target replication,
                        int to indicate how deep down the tree the target replication should be,
                        only works for LSTM at level 1, and for Tree-LSTM at any level
    """

    def __init__(self, model, batch_size, optimizer='adam', learning_rate=0.01,
                 weight_decay=0.0001, target_replication=None):
        super(Trainer, self).__init__()
        self.model = model
        if optimizer == 'adagrad':
            self.optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()),
                                           lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                       lr=learning_rate, weight_decay=weight_decay)
        else:
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=learning_rate, weight_decay=weight_decay)
        self.classes = model.label_dim
        self.batch_size = batch_size
        self.cuda_mode = self.model.cuda_mode
        self.embedding_dim = self.model.embedding_dim
        if self.cuda_mode:
            torch.backends.cudnn.benchmark = True
            self.model.cuda()
            self.criterion.cuda()
        # Determining model
        self.model_type = 0
        model_type = type(self.model)
        if model_type == StructureTreeLSTM:
            self.model_type = TREE_MODEL
        elif model_type == LSTMClassifier:
            self.model_type = LSTM_MODEL
        elif model_type == MLPClassifier:
            self.model_type = MLP_MODEL
        # Determining optimizer
        if type(target_replication) == int and target_replication > 0 and (self.model_type == TREE_MODEL or self.model_type == LSTM_MODEL):
            if self.model_type == TREE_MODEL:
                self.target_replication = target_replication
            if self.model_type == LSTM_MODEL:
                self.target_replication = 1
            if model.multilabel:
                self.criterion = CustomBCELoss()
            else:
                self.criterion = CustomCrossEntropyLoss()
        else:
            self.target_replication = None
            if model.multilabel:
                self.criterion = nn.BCELoss()
            else:
                self.criterion = nn.CrossEntropyLoss()

    def fit_generator(self, epoch_count, train_generator, train_len, val_generator, val_len, test_generator,
                      test_len, test_at_all_epochs=True, checkpoint_dir='', exp_name='experiment',
                      embed_model_file=None, embed_model_binary=None, word_model=True,
                      custom_metrics=dict(), macro_f1_score=True, micro_f1_score=True,
                      accuracy_score=True, all_f1_scores=False, auc_score=False, eval_metric='Macro-F1'):
        """
        Fit the model entirely.
        :param epoch_count: int, number of epochs for training
        :param train_generator: function with yield, generates training datapoints, 
                                in pairs (label, doctree) or (label, list (of lists ...) of strings)
        :param train_len: int, number of training datapoints
        :param val_generator: function with yield, generates validation datapoints,
                              in pairs (label, doctree) or (label, list (of lists ...) of strings)
        :param val_len: int, number of validation datapoints
        :param test_generator: function with yield, generates testing datapoints,
                               in pairs (label, doctree) or (label, list (of lists ...) of strings)
        :param test_len: int, number of testing datapoints
        :param test_at_all_epochs: bool, to test the train and test sets for all epochs even
                                   if the validation score is not the best achieved,
                                   can lengthen the training if True
        :param checkpoint_dir: str, path where the log file will be saved, the directory should be already created
        :param exp_name: str, experiment name, which is the name that the log file will have
        :param embed_model_file: str, file where to import the training embedding model from to get KeyedVectors model
        :param embed_model_binary: bool, indicates whether the model is binary,
                                   optional as .bin files are automatically tagged as binary
        :param word_model: bool, whether the given embedding model is a word embedding one
        :param custom_metrics: dictionary of {label of metric (str): metric (function)}, additional metrics to use
        :param macro_f1_score: bool, whether to compute the Macro-F1 score,
                               no need to include it in the custom_metrics
        :param micro_f1_score: bool, whether to compute the Micro-F1 score,
                               no need to include it in the custom_metrics
        :param accuracy_score: bool, whether to compute the accuracy score,
                               no need to include it in the custom_metrics
        :param all_f1_scores: bool, whether to compute the F1 scores of each class,
                              no need to include it in the custom_metrics
        :param auc_score: bool, whether to compute the ROC/AUC score,
                          no need to include it in the custom_metrics
        :param eval_metric: str, the metric to follow to decide which epoch gives the best score
        :return: None, saves checkpoint, displays logs and saves them
        """
        save_dir = os.path.join(checkpoint_dir, exp_name)
        # logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
        # file logger
        fh = logging.FileHandler(save_dir + '.log', mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # console logger
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # Loading Word2Vec model if available
        embedding_model = None
        if embed_model_file is not None:
            if embed_model_binary is None:
                embed_model_binary = embed_model_file.endswith('.bin')
            embedding_model = KeyedVectors.load_word2vec_format(embed_model_file, binary=embed_model_binary)
        # rest of the program
        best = -float('inf')
        for epoch in range(1, epoch_count + 1):
            self.epoch_fit_generator(epoch, train_generator, train_len, embedding_model=embedding_model)
            val_log, val_metrics = self.epoch_eval_generator(epoch, val_generator, val_len, 'val',
                                                             embedding_model=embedding_model,
                                                             word_model=word_model,
                                                             custom_metrics=custom_metrics,
                                                             macro_f1_score=macro_f1_score,
                                                             micro_f1_score=micro_f1_score,
                                                             accuracy_score=accuracy_score,
                                                             all_f1_scores=all_f1_scores,
                                                             auc_score=auc_score)
            logger.info(val_log)
            if best < val_metrics[eval_metric] or test_at_all_epochs:
                train_log, train_metrics = self.epoch_eval_generator(epoch, train_generator, train_len, 'train',
                                                                     embedding_model=embedding_model,
                                                                     word_model=word_model,
                                                                     custom_metrics=custom_metrics,
                                                                     macro_f1_score=macro_f1_score,
                                                                     micro_f1_score=micro_f1_score,
                                                                     accuracy_score=accuracy_score,
                                                                     all_f1_scores=all_f1_scores,
                                                                     auc_score=auc_score)
                logger.info(train_log)
                test_log, test_metrics = self.epoch_eval_generator(epoch, test_generator, test_len, 'test',
                                                                   embedding_model=embedding_model,
                                                                   word_model=word_model,
                                                                   custom_metrics=custom_metrics,
                                                                   macro_f1_score=macro_f1_score,
                                                                   micro_f1_score=micro_f1_score,
                                                                   accuracy_score=accuracy_score,
                                                                   all_f1_scores=all_f1_scores,
                                                                   auc_score=auc_score)
                logger.info(test_log)
                if best < val_metrics[eval_metric]:
                    best = val_metrics[eval_metric]
                    checkpoint = {
                        'model': self.model,
                        'optimizer': self.optimizer,
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics,
                        'test_matrics': test_metrics,
                        'epoch': epoch
                    }
                    logger.debug('==> New optimum found, checkpointing everything now...')
                    torch.save(checkpoint, save_dir + '.pt')

    def fit(self, epoch_count, train_data, val_data, test_data, test_at_all_epochs=True,
            checkpoint_dir='', exp_name='experiment', embed_model_file=None, embed_model_binary=None, word_model=True,
            custom_metrics=dict(), macro_f1_score=True, micro_f1_score=True, accuracy_score=True,
            all_f1_scores=False, auc_score=False, eval_metric='Macro-F1'):
        """
        Alias for fit_generator, works for a manageable number of datapoints.
        :param epoch_count: int, number of epochs for training
        :param train_data: pairs (label, doctree) or (label, list (of lists ...) of strings)
        :param val_data: pairs (label, doctree) or (label, list (of lists ...) of strings)
        :param test_data: pairs (label, doctree) or (label, list (of lists ...) of strings)
        :param test_at_all_epochs: bool, to test the train and test sets for all epochs even
                                   if the validation score is not the best achieved,
                                   can lengthen the training if True
        :param checkpoint_dir: str, path where the log file will be saved, the directory should be already created
        :param exp_name: str, experiment name, which is the name that the log file will have
        :param embed_model_file: str, file where to import the training embedding model from to get KeyedVectors model
        :param embed_model_binary: bool, indicates whether the model is binary,
                                   optional as .bin files are automatically tagged as binary
        :param word_model: bool, whether the given embedding model is a word embedding one
        :param custom_metrics: dictionary of {label of metric (str): metric (function)}, additional metrics to use
        :param macro_f1_score: bool, whether to compute the Macro-F1 score,
                               no need to include it in the custom_metrics
        :param micro_f1_score: bool, whether to compute the Micro-F1 score,
                               no need to include it in the custom_metrics
        :param accuracy_score: bool, whether to compute the accuracy score,
                               no need to include it in the custom_metrics
        :param all_f1_scores: bool, whether to compute the F1 scores of each class,
                              no need to include it in the custom_metrics
        :param auc_score: bool, whether to compute the ROC/AUC score,
                          no need to include it in the custom_metrics
        :param eval_metric: str, the metric to follow to decide which epoch gives the best score
        :return: None, saves checkpoint, displays logs and saves them
        """
        return self.fit_generator(epoch_count, cycle(train_data), len(train_data), cycle(val_data), len(val_data),
                                  cycle(test_data), len(test_data), test_at_all_epochs=test_at_all_epochs,
                                  checkpoint_dir=checkpoint_dir, exp_name=exp_name,
                                  embed_model_file=embed_model_file, embed_model_binary=embed_model_binary,
                                  word_model=word_model,
                                  custom_metrics=custom_metrics, macro_f1_score=macro_f1_score,
                                  micro_f1_score=micro_f1_score, accuracy_score=accuracy_score,
                                  all_f1_scores=all_f1_scores, auc_score=auc_score,
                                  eval_metric=eval_metric)

    def fit_and_split(self, epoch_count, data, test_at_all_epochs=True, checkpoint_dir='',
                      exp_name='experiment', embed_model_file=None, embed_model_binary=None, word_model=True,
                      train_size=0.8, test_val_split=0.5,
                      custom_metrics=dict(), macro_f1_score=True, micro_f1_score=True,
                      accuracy_score=True, all_f1_scores=False, auc_score=False, eval_metric='Macro-F1'):
        """
        Alias for fit, splits datapoints before fitting the model.
        :param epoch_count: int, number of epochs for training
        :param data: pairs (label, doctree) or (label, list (of lists ...) of strings)
        :param test_at_all_epochs: bool, to test the train and test sets for all epochs even
                                   if the validation score is not the best achieved,
                                   can lengthen the training if True
        :param checkpoint_dir: str, path where the log file will be saved, the directory should be already created
        :param exp_name: str, experiment name, which is the name that the log file will have
        :param embed_model_file: str, file where to import the training embedding model from to get KeyedVectors model
        :param embed_model_binary: bool, indicates whether the model is binary,
                                   optional as .bin files are automatically tagged as binary
        :param word_model: bool, whether the given embedding model is a word embedding one
        :param train_size: float, determines the size of the training set
        :param test_val_split: float, percentage that the test set should take of the non-train data
        :param custom_metrics: dictionary of {label of metric (str): metric (function)}, additional metrics to use
        :param macro_f1_score: bool, whether to compute the Macro-F1 score,
                               no need to include it in the custom_metrics
        :param micro_f1_score: bool, whether to compute the Micro-F1 score,
                               no need to include it in the custom_metrics
        :param accuracy_score: bool, whether to compute the accuracy score,
                               no need to include it in the custom_metrics
        :param all_f1_scores: bool, whether to compute the F1 scores of each class,
                              no need to include it in the custom_metrics
        :param auc_score: bool, whether to compute the ROC/AUC score,
                          no need to include it in the custom_metrics
        :param eval_metric: str, the metric to follow to decide which epoch gives the best score
        :return: None, saves checkpoint, displays logs and saves them
        """
        train_data, test_data = train_test_split(data, test_size=1-train_size)
        val_data, test_data = train_test_split(test_data, test_size=test_val_split)
        return self.fit(epoch_count, train_data, val_data, test_data,
                        test_at_all_epochs=test_at_all_epochs, checkpoint_dir=checkpoint_dir, exp_name=exp_name,
                        embed_model_file=embed_model_file, embed_model_binary=embed_model_binary, word_model=word_model,
                        custom_metrics=custom_metrics, macro_f1_score=macro_f1_score, micro_f1_score=micro_f1_score,
                        accuracy_score=accuracy_score, all_f1_scores=all_f1_scores,
                        auc_score=auc_score, eval_metric=eval_metric)

    def epoch_fit_generator(self, epoch, generator, steps_per_epoch, test=False, embedding_model=None, word_model=True):
        """
        Fitting for one epoch with a generator.
        :param epoch: int, the number of the epoch
        :param generator: function with yield, generates datapoints,
                          in pairs (label, doctree) or (label, list (of lists ...) of strings)
        :param steps_per_epoch: number of steps this epoch has
        :param test: bool, whether this epoch is for testing/validating (True) or training (False)
        :param embedding_model: gensim KeyedVectors, pre-trained embedding model, to embed text if not done
        :param word_model: bool, whether the given embedding model is a word embedding one
        :return: average loss, and if not test, the true and predicted labels
        """
        if test:
            self.model.eval()
        else:
            self.model.train()
        total_loss = 0
        doc_count = 0
        predictions = []
        true_labels = []
        targets = []
        outputs = []
        train_count = 0
        mode = 'train' if not test else 'test'
        for _ in tqdm(range(steps_per_epoch), desc='{} epoch {}'.format(mode, str(epoch))):
            y, X = next(generator)
            if embedding_model is not None:
                if self.model_type == TREE_MODEL:
                    X = form_doctree_by_dfs(X, self.embedding_dim, embedding_model=embedding_model)
                elif self.model_type == LSTM_MODEL:
                    X = form_embedding_sequence(X, self.embedding_dim, embedding_model, word_model)
                elif self.model_type == MLP_MODEL:
                    X = np.mean(form_embedding_sequence(X, self.embedding_dim, embedding_model, word_model), axis=0)
            if not type(X) == DocTreeNode and self.model_type == TREE_MODEL:
                X = form_doctree_by_dfs(X, self.embedding_dim)
            doc_count += 1
            targets.append(y)
            output = self.model(X)
            if self.target_replication is not None:
                children_outputs = []
                children_nodes = X.children
                target_replication_level = 0
                while target_replication_level < self.target_replication and children_nodes:
                    children_outputs += [torch.stack([self.model(child)]) for child in children_nodes]
                    target_replication_level += 1
                    children_nodes = [sub_child for child in children_nodes for sub_child in child.children]
                outputs.append((torch.stack([output]), children_outputs))
            else:
                outputs.append(output)
            train_count += 1
            if test:
                predictions.append(self.model.get_prediction(output))
                true_labels.append(y)
            train_count += 1
            if train_count == self.batch_size:
                target_var = Variable(torch.LongTensor(np.array(targets)))
                if self.cuda_mode:
                    target_var = target_var.cuda()
                if self.target_replication is None:
                    outputs = torch.stack(outputs)
                batch_loss = self.criterion(outputs, target_var)
                total_loss += batch_loss.data[0]
                if not test:
                    batch_loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                targets = []
                outputs = []
                train_count = 0
        if train_count > 0:
            target_var = Variable(torch.LongTensor(np.array(targets)))
            if self.cuda_mode:
                target_var = target_var.cuda()
            if self.target_replication is None:
                outputs = torch.stack(outputs)
            batch_loss = self.criterion(outputs, target_var)
            total_loss += batch_loss.data[0]
            if not test:
                batch_loss.backward()
                self.optimizer.step()
        if not test:
            return total_loss / doc_count
        else:
            return [total_loss / doc_count, np.array(true_labels), np.array(predictions)]

    def epoch_fit(self, epoch, dataset, embedding_model=None, word_model=True):
        """
        Alias for epoch_fit_generator, when the number of datapoints is manageable.
        :param epoch: int, the number of the epoch
        :param dataset: pairs (label, doctree) or (label, list (of lists ...) of strings)
        :param embedding_model: gensim KeyedVectors, pre-trained embedding model, to embed text if not done
        :param word_model: bool, whether the given embedding model is a word embedding one
        :return: average loss, and if not test, the true and predicted labels
        """
        return self.epoch_fit_generator(epoch, iter(dataset), len(dataset),
                                        embedding_model=embedding_model, word_model=word_model)

    def epoch_test(self, epoch, dataset, embedding_model=None, word_model=True):
        """
        Alias for epoch_fit_generator, when the number of datapoints is manageable and are for testing.
        :param epoch: int, the number of the epoch
        :param dataset: pairs (label, doctree) or (label, list (of lists ...) of strings)
        :param embedding_model: gensim KeyedVectors, pre-trained embedding model, to embed text if not done
        :param word_model: bool, whether the given embedding model is a word embedding one
        :return: average loss, and if not test, the true and predicted labels
        """
        return self.epoch_fit_generator(epoch, iter(dataset), len(dataset), test=True,
                                        embedding_model=embedding_model, word_model=word_model)

    def epoch_test_generator(self, epoch, generator, steps_per_epoch, embedding_model=None, word_model=True):
        """
        Alias for epoch_fit_generator, for the case of testing.
        :param epoch: int, the number of the epoch
        :param generator: function with yield, generates datapoints,
                          in pairs (label, doctree) or (label, list (of lists ...) of strings)
        :param steps_per_epoch: number of steps this epoch has
        :param embedding_model: gensim KeyedVectors, pre-trained embedding model, to embed text if not done
        :param word_model: bool, whether the given embedding model is a word embedding one
        :return: average loss, and if not test, the true and predicted labels
        """
        return self.epoch_fit_generator(epoch, generator, steps_per_epoch, test=True,
                                        embedding_model=embedding_model, word_model=word_model)

    def epoch_eval_generator(self, epoch, generator, steps_per_epoch, mode, embedding_model=None, word_model=True,
                             custom_metrics=dict(), macro_f1_score=True, micro_f1_score=True,
                             accuracy_score=True, all_f1_scores=False, auc_score=False):
        """
        Evaluates an epoch using a generator
        :param epoch: int, the number of the epoch
        :param generator: function with yield, generates datapoints,
                          in pairs (label, doctree) or (label, list (of lists ...) of strings)
        :param steps_per_epoch: number of steps this epoch has
        :param mode: str, the name of the mode ('train': training, 'test': testing, 'val': validation)
        :param embedding_model: gensim KeyedVectors, pre-trained embedding model, to embed text if not done
        :param word_model: bool, whether the given embedding model is a word embedding one
        :param custom_metrics: dictionary of {label of metric (str): metric (function)}, additional metrics to use
        :param macro_f1_score: bool, whether to compute the Macro-F1 score,
                               no need to include it in the custom_metrics
        :param micro_f1_score: bool, whether to compute the Micro-F1 score,
                               no need to include it in the custom_metrics
        :param accuracy_score: bool, whether to compute the accuracy score,
                               no need to include it in the custom_metrics
        :param all_f1_scores: bool, whether to compute the F1 scores of each class,
                              no need to include it in the custom_metrics
        :param auc_score: bool, whether to compute the ROC/AUC score,
                          no need to include it in the custom_metrics
        :return: log (string), metrics (dictionary of all metrics (label of metric, metric))
        """
        loss, y_true, y_pred = self.epoch_test_generator(epoch, generator, steps_per_epoch,
                                                         embedding_model=embedding_model, word_model=word_model)
        if macro_f1_score:
            custom_metrics['Macro-F1'] = macro_f1
        if micro_f1_score:
            custom_metrics['Micro-F1'] = micro_f1
        if accuracy_score:
            custom_metrics['Accuracy'] = accuracy
        if all_f1_scores:
            custom_metrics['All F1s'] = all_f1s
        if auc_score:
            custom_metrics['ROC/AUC'] = roc_auc_score
        metrics = {}
        log = '{}\t==> Epoch {} \tLoss: {}'.format(mode, epoch, loss)
        for metric in custom_metrics:
            metrics[metric] = custom_metrics[metric](y_true, y_pred)
            log += '\t{}: {}'.format(metric, metrics[metric])
        return log, metrics