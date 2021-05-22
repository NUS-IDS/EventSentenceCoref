#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 21:23:37 2020

@author: sdas
"""

from tqdm import tqdm
import numpy as np

import torch
import pickle



UNK_TOKEN = "unknown"

def load_glove_dict(embedding_file):
    termdict = dict()
    lines = open(embedding_file, "r", encoding="utf-8").readlines()
    for line in tqdm(lines):
        word_vec = line.split(" ")
        word = word_vec[0]
        termdict[word] = 0   
    
    print ("#words in embeddings file "+str(len(termdict)))
    return termdict

def load_glove_embedding(embedding_file, mydict, emdim):
    word2embedding = dict()
    lines = open(embedding_file, "r", encoding="utf-8").readlines()
    for line in tqdm(lines):
        word_vec = line.split(" ")
        word = word_vec[0]
        vec = np.array(word_vec[1:], dtype=np.float32)
        word2embedding[word] = vec
    embedding = np.zeros((len(mydict), emdim), dtype=np.float32)
    num_oov = 0
    for word, idx in mydict.items():
        if word in word2embedding:
            embedding[idx] = word2embedding[word]
        else:
            embedding[idx] = word2embedding[UNK_TOKEN]
            num_oov += 1
    print("num OOV : {}".format(num_oov))
    
    return embedding

def make_embedding(embedding_file, output_file, mydict, emdim):
    word2embedding = dict()
    lines = open(embedding_file, "r", encoding="utf-8").readlines()
    for line in tqdm(lines):
        word_vec = line.split(" ")
        word = word_vec[0]
        vec = np.array(word_vec[1:], dtype=np.float32)
        word2embedding[word] = vec
    embedding = np.zeros((len(mydict), emdim), dtype=np.float32)
    num_oov = 0
    for word, idx in mydict.items():
        if word in word2embedding:
            embedding[idx] = word2embedding[word]
        else:
            embedding[idx] = word2embedding[UNK_TOKEN]
            num_oov += 1
    print("num OOV : {}".format(num_oov))
    with open(output_file, "wb") as f:
        pickle.dump(embedding, f)
    return embedding


def load_embedding(inppklfile):
    
    with open(inppklfile, "rb") as f:
        embedding = pickle.load(f)
        embedding = torch.tensor(embedding,
                                 dtype=torch.float)

    f.close()
    return embedding
    

