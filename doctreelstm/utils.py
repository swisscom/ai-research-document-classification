"""
Copyright © 2018-present, Swisscom (Schweiz) AG.
All rights reserved.

Author: Khalil Mrini
"""

from doctreelstm.doctree import DocTreeNode
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def form_doctree_by_dfs(doc, embedding_dim, compute_hierarchical_avg=False, embedding_model=None, text_doc=None):
    """
    Forming document tree by depth-first search.
    Recursive function.
    :param doc: list of (lists of lists of etc.) vectors (real numbers)
                or strings (words or sentences, depending on the embedding model)
    :param embedding_dim: int, to comply with the embedding dimension given
    :param compute_hierarchical_avg: bool, indicates if embedding should be computed as hierarchical average of children
    :param embedding_model: gensim KeyedVectors, the pre-trained embedding model.
                            If None, the embedding model is not used.
    :param text_doc: list of (lists of lists of etc.) strings (words or sentences) that match the tree of embeddings
    :return: DocTreeNode if valid, else None
    """
    if type(doc) == list and len(doc) > 0:
        node = DocTreeNode()
        child_count = 0
        if type(doc[0]) == list:
            for child_idx in range(len(doc)):
                child = doc[child_idx]
                child_text = None
                if text_doc is not None:
                    child_text = text_doc[child_idx]
                child_node = form_doctree_by_dfs(child, embedding_dim, compute_hierarchical_avg=compute_hierarchical_avg, embedding_model=embedding_model, text_doc=child_text)
                if child_node is not None:
                    node.add_child(child_node)
                    child_count += 1
            if child_count > 0:
                if compute_hierarchical_avg:
                    embedding = node.average_simply()
                    if embedding.size == embedding_dim:
                        node.embedding = embedding
                return node
        elif type(doc[0]) == float and len(doc) == embedding_dim:
            return DocTreeNode(embedding=doc, is_avg=False, text=text_doc)
        elif type(doc[0]) == str and embedding_model is not None:
            for child_text in doc:
                child_vector = None
                try:
                    child_vector = embedding_model[child_text]
                except:
                    try:
                        child_vector = embedding_model[child_text.lower()]
                        child_text = child_text.lower()
                    except:
                        pass
                if child_vector is not None:
                    node.add_child(DocTreeNode(embedding=child_vector, is_avg=False, text=child_text))
                    child_count += 1
            if child_count > 0:
                if compute_hierarchical_avg:
                    embedding = node.average_simply()
                    if embedding.size == embedding_dim:
                        node.embedding = embedding
                return node
    return None

def form_embedding_sequence(document, embedding_dim, embedding_model, word_model):
    """
    Form a sequence of embeddings for the LSTM and MLP Classifiers.
    :param document: list of sentences, that are list of strings (sentence embedding model)
                     or list of lists of strings (word embedding model)
    :param embedding_dim: int, embedding dimension of the embedding model
    :param embedding_model: gensim KeyedVectors, the pre-trained embedding model.
                            If None, the embedding model is not used.
    :param word_model: bool, whether the given embedding model is a word embedding one
    :return: list of numpy arrays
    """
    vec = []
    for sentence in document:
        sent_vec = []
        if word_model:
            for word in sentence:
                try:
                    sent_vec.append(embedding_model[word])
                except:
                    try:
                        sent_vec.append(embedding_model[word.lower()])
                    except:
                        pass
            if sent_vec:
                sent_vec = np.mean(sent_vec, axis=0)
        else:
            try:
                sent_vec = embedding_model[sentence]
            except:
                pass
        if not sent_vec:
            sent_vec = np.zeros(embedding_dim)
    vec.append(sent_vec)
    return vec

### METRICS ###

def accuracy(y_true, y_pred):
    """
    Accuracy score.
    :param y_true: true labels, list of int
    :param y_pred: predicted labels, list of int
    :return: float
    """
    return accuracy_score(np.ndarray.flatten(y_true), np.ndarray.flatten(y_pred))

def macro_f1(y_true, y_pred):
    """
    Macro-F1 score.
    :param y_true: true labels, list of int
    :param y_pred: predicted labels, list of int
    :return: float
    """
    return f1_score(y_true, y_pred, average='macro')

def micro_f1(y_true, y_pred):
    """
    Micro-F1 score.
    :param y_true: true labels, list of int
    :param y_pred: predicted labels, list of int
    :return: float
    """
    return f1_score(y_true, y_pred, average='micro')

def all_f1s(y_true, y_pred):
    """
    All F1 scores of each class, in order (int increasing order).
    :param y_true: true labels, list of int
    :param y_pred: predicted labels, list of int
    :return: list of float
    """
    return f1_score(y_true, y_pred, average=None)