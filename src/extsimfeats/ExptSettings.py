#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 13:37:33 2021

@author: sdas
"""

rseed=3001983
OOV_WORD='oov'
weFile="/home/sdas/embeddings/glove.6B.200d.txt"
weDim=200
tthresh=0

stopwordsf="/home/sdas/setups/mallet-2.0.8/stoplists/en.txt"

def loadStopwords():
    
    lines = open(stopwordsf, "r").readlines()
    stopwords={}
    for line in lines:
        stopwords[line.strip()]=""
        
    return stopwords
    