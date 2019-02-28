# Document Tree-LSTM (`doctreelstm`)

**Author:** Khalil Mrini

Copyright © 2018-present, Swisscom (Schweiz) AG. \
All rights reserved.

## 1. Description

This package is the implementation of Tree-LSTMs for structured 
documents, also called *Structure Tree-LSTM*, or *Document Tree-LSTM*.
The package can embed documents hierarchically with a given
pre-trained embedding model, convert a document organised as lists
of (lists of...) text or vectors to a tree, and train a classifier 
model based on *PyTorch*. The package is entirely modular, and 
requires no particular resources to download.

## 2. Installation

After cloning this repository, go to the root directory and run 
the following command:
```python
pip -r requirements.txt
```

## 3. Data Format

Any list of samples as training, validation or test set should be 
a list of pairs (*label*, *document*), **NOT** one pair of two lists.

## 4. Use Cases

### 4.1. Import

The only import for this package that is needed is the following:

```python
from doctreelstm import *
```

This imports:
1. from `models.py`: `StructureTreeLSTM`, `LSTMClassifier`, `MLPClassifier`
2. from `trainer.py`: `Trainer`
3. from `doctree.py`: `DocTreeNode`
4. from `utils.py`: `form_doctree_by_dfs`, `form_embedding_sequence`

It is advised to look at these files and read about the different
arguments these classes and functions take.

### 4.2. Training the model with embedded data

Here, the embedded data is saved into a `data.pkl` file, which 
contains a dictionary of lists of samples for training, validation,
and testing. We load the file:

```python
import pickle
emb_docs = pickle.load(open('data.pkl', 'rb'))
```

We initiate a model and its trainer, and directly train for 
100 epochs:

```python
from doctreelstm import *
model = StructureTreeLSTM(700, 128, 4, False)
trainer = Trainer(model, 32, target_replication=1)
trainer.fit(100, emb_docs['train'], emb_docs['val'], 
    emb_docs['test'])
```

In the above, `StructureTreeLSTM` can be replaced by 
`LSTMClassifier` or `MLPClassifier`.

### 4.3. Embedding the data while training

Given a pre-trained sentence embedding model saved in a binary file 
`model.bin` and to be used in a `gensim` `KeyedVectors` instance, 
it is possible to embed on the go:

```python
trainer.fit(100, emb_docs['train'], emb_docs['val'], 
    emb_docs['test'], embed_model_file='model.bin', 
    embed_model_binary=True, word_model=False)
```

### 4.4. Training with a large number of samples

In the case of training with a number of samples too large to handle,
it is advised to use a generator to replace the datasets. The number
of samples has to be known in advance:

```python
trainer.fit_generator(100, train_generator, 800, 
    val_generator, 100, test_generator, 100)
```

### 4.5. Predicting the class for samples

Given a list of samples and a trained `Model` instance, we can
predict using the following code:

```python
predictions = model.predict(samples)
```

### 4.6. Determining the weights of units in the document tree

A function that is only available for the Structure Tree-LSTM model
enables to predict the class for one label, and give the weights
that each children unit have in the prediction:

```python
prediction, weight_list = model.predict_with_weights(doctree)
```