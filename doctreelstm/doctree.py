"""
Copyright © 2018-present, Swisscom (Schweiz) AG.
All rights reserved.

Author: Khalil Mrini
"""

import numpy as np

class DocTreeNode(object):
    """
    Node class forming a document tree by recursion.
    It contains a vector and a list of children nodes.

    Parameters
    ----------
    embedding: numpy ndarray (if actual embedding) or None (by default, when no embedding)
    is_avg: bool, determines if node has genuine embedding or embedding computed as average
    text: str or None, is the text corresponding to the embedding
    """

    def __init__(self, embedding=None, is_avg=True, text=None):
        self.embedding = embedding
        if self.embedding is not None and not type(self.embedding) == np.ndarray:
            self.embedding = np.array(self.embedding)
        self.is_avg = is_avg
        self.text = text
        self.children = []
        self.num_children = 0

    def add_child(self, child):
        """
        Adds a node to the list of children of the current one.
        :param child: DocTreeNode
        """
        self.num_children += 1
        self.children.append(child)

    def collect_children(self):
        """
        Collects children recursively using the function given as argument.
        :return: numpy ndarray, array of arrays of float
        """
        input_list = []
        if not self.is_avg:
            input_list.append(self.embedding)
        if self.is_avg and self.embedding is not None:
            input_list.append(self.embedding)
        else:
            for child_node in self.children:
                child_input = child_node.average_simply()
                input_list.append(child_input)
        return np.array(input_list)

    def average_simply(self):
        """
        Wraps the tree (aggregates its value) by giving an averaged vector per node.
        Averages each node individually and ignores nodes having an empty list as vector.
        Recursive function.
        :return: numpy.ndarray
        """
        children_inputs = self.collect_children()
        if not children_inputs.size == 0:
            return np.mean(children_inputs, axis=0)
        else:
            return np.empty(0)