#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 23:36:42 2021

@author: sdas
"""
import json
from random import randint, seed
import regex as re
import numpy as np
import ExptSettings as settings
import gloveLoader as gl
import pickle
import stanza
nlp = stanza.Pipeline(lang='en', use_gpu=False, processors='tokenize,pos,lemma,depparse,ner')

#nlp = stanza.Pipeline(lang='pt', processors='tokenize,pos,lemma,depparse')
#use markNEREntities2 call for Portuguese

def markNEREntities2(taggeddoc):
    
    allnertags= []
    for taggedsent in taggeddoc.sentences:
        sentwords = taggedsent.words
        nertags = []
        for word in sentwords:
            nertags.append('O')
        
        allnertags.extend(nertags)
    
    return allnertags


def markNEREntities(taggeddoc, entities):
    
    allnertags= []
    for taggedsent in taggeddoc.sentences:
        sentwords = taggedsent.words
        nertags = []
        for word in sentwords:
            nertags.append('O')
            
        for entity in entities:
            
             etype = entity.type
             entwords = entity.text.strip().split()
        
             for wx, word in enumerate(sentwords):
                match=False
                if word.text==entwords[0]:
                    match=True
                    
                    if (wx+len(entwords)) >= len(sentwords):
                        match=False
                        break
                    
                    for ex in range(1, len(entwords)):                    
                        if entwords[ex]!=sentwords[wx+ex].text:
                            match=False
                            break
                    
                            
                if match:
                    for ex in range(0, len(entwords)):
                        nertags[wx+ex]=etype
   
        allnertags.extend(nertags)
    
    return allnertags

def getPOSText(taggeddoc):

    postokens = []
    texttokens = []
    
    for taggedsent in taggeddoc.sentences:
        for token in taggedsent.words:
            texttokens.append(token.text)
            postokens.append(token.upos)
            
    return postokens, texttokens


def get_dependency_parse_bigrams(taggeddoc):
    dbigs=[]
    for sent in taggeddoc.sentences:
        for word in sent.words:
            if word.head ==0:
                head="root"
            else:
                head = sent.words[word.head-1].text
            
            dbigs.append((word.text, word.deprel, head))
   
    return dbigs

def loadTrainJSON(inpfiles):
    
    totalarticles = 0
    iid2pairs={}
    iid2clusters={}
    iid2sentmap={}
    totpairs = 0
    for inpfile in inpfiles:
        print ("Processing "+inpfile)
        with open(inpfile, "r") as fin:
            
            lx = 0
            
            for line in fin.readlines():
                
                lx += 1
                
                data = json.loads(line)
                if 'url' in data:
                    iid =  (data['url'])
                elif 'id' in data:
                    iid = str(data['id'])
                    
                totalarticles += 1        
                sentences = data['sentences']
                
                if 'event_clusters' not in data:
                    event_clusters = []
                else:
                    event_clusters = data['event_clusters']
                
                pos_sentences = data['sentence_no']
                
    #            print ("\n#sentences "+str(len(sentences)))
    #            print ("#pos_sentences "+str(len(pos_sentences)))
    #            print ("#clusters "+str(len(event_clusters)))
                
                sent_map={}
                
                pairs=[]
                for sx, sentence in enumerate(sentences):
                    
                    sentnum = pos_sentences[sx]
                    sent_map[sentnum]=sentence
                
                    for sx2 in range (sx+1, len(sentences)):
                        pairs.append((sentnum, pos_sentences[sx2]))
                        
                iid2pairs[iid] = pairs
                iid2sentmap[iid] = sent_map
                totpairs += len(pairs)
                sent2cluster = {}
                for cx, cluster in enumerate(event_clusters):
                    #print ("\nCluster "+str(cx)+" "+str(cluster))
                    for sentnum in cluster:
                        sent2cluster[sentnum] = cx    
            
                iid2clusters[iid] = sent2cluster
            
            
        print ("len(iid2clusters) "+str(len(iid2clusters))+ \
               " len(iid2pairs) "+str(len(iid2pairs)))
        print ("#pairs "+str(totpairs))
        
    return iid2pairs, iid2clusters, iid2sentmap


def splitTrainVal(data, rseed, nfolds):
    
    seed(rseed)
    val={}
    train={}
    
    for iid in data:
        
        opt = randint(0, nfolds)
        if opt==0:
            val[iid] = data[iid]
        else:
            train[iid] = data[iid]
        
        
    return train, val

def normalize(sentence):
    
    taggeddoc = nlp(sentence)
    postags, words = getPOSText(taggeddoc)
    nertags = markNEREntities(taggeddoc, taggeddoc.entities)
    #nertags = markNEREntities2(taggeddoc)
    dbigs = get_dependency_parse_bigrams(taggeddoc)
    return words, postags, nertags, dbigs



def loadSerializedData(inpProcDir):

    iid2pairs = pickle.load(open(inpProcDir+"/iid2pairs.pkl", "rb"))
    iid2clustermap = pickle.load(open(inpProcDir +"/iid2clustermap.pkl", "rb"))
    iid2sentmap = pickle.load(open(inpProcDir+"/iid2sentmap.pkl", "rb"))
    iid2procsentmap = pickle.load(open(inpProcDir+"/iid2procsentmap.pkl", "rb"))
    
#    temp1 = pickle.load(open("/home/sdas/fiona/to_suj/val_bertprobs.pickle", "rb"))
#    temp2 = pickle.load(open("/home/sdas/fiona/to_suj/train_bertprobs.pickle", "rb"))
    
    iid2similarities = {}
#    for iid in (temp1):
#        iid2similarities[iid] = temp1[iid]
#    for iid in (temp2):
#        iid2similarities[iid] = temp2[iid]
    
    #iid2similarities = pickle.load(open(inpProcDir+"/bertsim.pkl", "rb"))
    
    return iid2pairs, iid2clustermap, iid2sentmap, iid2procsentmap, iid2similarities
    
def createTrainDictionary(tthresh, iid2pairs2consider, iid2procsentmap):
    
    stopwords = settings.loadStopwords()
    
    print ("#stopwords "+str(len(stopwords)))
    
    termcounts = {}
    termdictionary = {}    
    termdictionary_r = {}    
    totterms = 0
    aggdoccounts = {}
    for iid in iid2pairs2consider:
        doccounts = {}
        
        sentmap = iid2procsentmap[iid]
        
        for sentx in sentmap:
            
            words, _, _, _ = sentmap[sentx]
            
            for word in words:
                word = word.lower()
                if word in stopwords:
                    continue
                if word not in termdictionary:
                    wordid = len(termdictionary)
                    termdictionary[word] = wordid
                    termdictionary_r[wordid] = word
                    termcounts [wordid] = 0
                
                doccounts[wordid] = 1
                wordid = termdictionary[word]
                termcounts [wordid] += 1
                totterms += 1
            
           
        for wordid in doccounts:
            if wordid not in aggdoccounts:
                aggdoccounts[wordid] = 1
            else:
                aggdoccounts[wordid] += 1
            
    totdoccount = len(iid2pairs2consider)
    ntermdict={}
    ntermdict_r={}
    idftable={}
    print ("#terms before filter "+str(len(termdictionary)))    
    
    
           
    for wordid in termcounts:
        if termcounts[wordid] > tthresh:
            nwordid = len(ntermdict)
            word = termdictionary_r[wordid]
            ntermdict[nwordid] = word
            ntermdict_r[word] = nwordid
            
            idftable[word] = np.log2(totdoccount/(aggdoccounts[wordid]+1))
    
    oovid = len(ntermdict)
    ntermdict_r[settings.OOV_WORD] = oovid
    ntermdict[oovid] = settings.OOV_WORD
    idftable[settings.OOV_WORD] = 1 
    
    return ntermdict, ntermdict_r, idftable
      
def normalize2(sentence, termmap):
    
    
    words = sentence.lower().split()
    newsent = ""
    for word in words:
        if word in termmap:
            newsent += " "+word
    
    return newsent

def createPairs(iid2pairs, iid2procsentmap, iid2clustermap, \
                termmap, iid2similarities):
    
    pospairs = []
    negpairs = []
    
    for iid in iid2pairs:
        
        pairs = iid2pairs[iid]
        sentmap = iid2procsentmap[iid]
        clustermap = iid2clustermap[iid]
        simmap = {}
        
        if iid in iid2similarities:
            simmap = iid2similarities[iid]
        
        for px in range(len(pairs)):
            
            (sent1x, sent2x) = pairs[px]
            wds1, pos1, ner1, dbigs1 = sentmap[sent1x]
            wds2, pos2, ner2, dbigs2 = sentmap[sent2x]
           
            lkupkey = str(sent1x)+" "+str(sent2x)
            if lkupkey not in simmap:
                lkupkey = str(sent2x)+" "+str(sent1x)
            
            pairsim = 0
            if lkupkey in simmap:
                pairsim = simmap[lkupkey]
            
            c1 = clustermap[sent1x]
            c2 = clustermap[sent2x]
            
            if c1==c2:
                pospairs.append(((wds1, pos1, ner1, dbigs1), \
                                 (wds2, pos2, ner2, dbigs2), pairsim))
            else:
                negpairs.append(((wds1, pos1, ner1, dbigs1), \
                                 (wds2, pos2, ner2, dbigs2), pairsim))
                
    return pospairs, negpairs


   
    



def serializeData(inpfileslist, outdir):

    iid2pairs, iid2clustermap, iid2sentmap = \
        loadTrainJSON(inpfileslist)
    
    
    iid2procsentmap = {}
   
    for ix, iid in enumerate(iid2sentmap):
        
       
        sentmap = iid2sentmap[iid]
        procsentmap = {}
        for sx, sentnum in enumerate(sentmap):
            sentence = sentmap[sentnum]
            words, postags, nertags, dbigs = normalize(sentence)
            
            if ix%50==0 and sx==0:
                print ("Processing "+str(ix))
                print (words)
                print (postags)
                print (nertags)
                print (dbigs)
            
            procsentmap[sentnum] = (words, postags, nertags, dbigs)
        
        iid2procsentmap[iid] = procsentmap
       
        
#        
    pickle.dump(iid2pairs, open(outdir+"/iid2pairs.pkl", "wb"))
    pickle.dump(iid2clustermap, open(outdir+"/iid2clustermap.pkl", "wb"))
    pickle.dump(iid2sentmap, open(outdir+"/iid2sentmap.pkl", "wb"))
    pickle.dump(iid2procsentmap, open(outdir+"/iid2procsentmap.pkl", "wb"))
#        
    
def getUniquePOSNERTags(iid2procsentmap):
    
    uniqueNERTags={}
    uniquePOSTags={}
    
    for iid in iid2procsentmap:
        
        sentmap = iid2procsentmap[iid]
        for sentnum in sentmap:
            
            (words, postags, nertags, crftags) = sentmap[sentnum]
            
            for postag in postags:
                if postag not in uniquePOSTags:
                    uniquePOSTags[postag]=""
                    
            for nertag in nertags:
                if nertag not in uniqueNERTags:
                    uniqueNERTags[nertag]=""
    
    print (len(uniqueNERTags))
    print (len(uniquePOSTags))
    

    
    dlist=['O', 'CARDINAL', 'DATE', 'TIME', 'PERCENT', \
           'ORDINAL', 'QUANTITY', 'MONEY']
 
    for tag in dlist:
        if tag in uniqueNERTags:
            del uniqueNERTags[tag]
    
    dlist=['AUX', 'CCONJ', 'SCONJ', 'PART', 'SYM', 'INTJ', 'ADP', \
          'PRON',  'PUNCT', 'DET' ]
    
    for tag in dlist:
        if tag in uniquePOSTags:
            del uniquePOSTags[tag]
    
    
    return uniquePOSTags, uniqueNERTags



