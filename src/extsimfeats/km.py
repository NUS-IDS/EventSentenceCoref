#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:32:52 2021

@author: sdas
"""

import ExptSettings as settings
import numpy as np





#####################################
def convert_one_instance(words, wdict):

    intseq=[]
    
    for word in words:
        word = word.lower()
        if word in wdict:
            intseq.append(wdict[word])
        else:
            intseq.append(wdict[settings.OOV_WORD])
            
    
    return (intseq)

#####################################
def getEmbdgDotProduct(wds1, wds2, wdict, emdgMatrix):
    
  

    seq1 = convert_one_instance(wds1, wdict)
    seq2 = convert_one_instance(wds2, wdict)
    
    dp = 0
    
    for wordid1 in seq1:
        wvec1 = emdgMatrix[wordid1]
        den1 = np.linalg.norm(wvec1, 2)
        
        for wordid2 in seq2:
            wvec2 = emdgMatrix[wordid2]
            den2 = np.linalg.norm(wvec2, 2)
            dp += np.dot(wvec1, wvec2)/(den1*den2)        

    return dp/(len(seq1) * len(seq2))
 
#####################################

def typeBasedSimilarity(tagtype, words1, tags1, words2, tags2):
    
    temp1 = {}
    for wx, word in enumerate(words1):
        word = word.lower()
        if tags1[wx]==tagtype:
            temp1[word]=1
    temp2 = {}
    for wx, word in enumerate(words2):
        word = word.lower()
        if tags2[wx]==tagtype:
            temp2[word]=1
    
    
    num = 0
    den1 = len(temp1)
    den2 = len(temp2)
    
    for word in temp1:
        if word in temp2:
            num += 1
        
    if den1==0 or den2==0:
        return 0
    else:
        return num/(np.sqrt(den1*den2))
    
#####################################   
    
def getTFIDFCosine(wds1, wds2, wdict, idftable):
    
    num = 0
    den1 = 0
    den2 = 0
    temp1={}
    for word in wds1:
        word = word.lower()
        if word not in temp1:
            temp1[word]= 1
        else:
            temp1[word]+= 1
    
    temp2={}
    for word in wds2:
        word = word.lower()
        if word not in temp2:
            temp2[word]= 1
        else:
            temp2[word]+= 1
    
    for word in temp1:
        
        if word in idftable:
            den1 += (temp1[word]*idftable[word])*(temp1[word]*idftable[word])
        else:
            den1 += \
            (temp1[word]*idftable[settings.OOV_WORD])*(temp1[word]*idftable[settings.OOV_WORD])
    
    for word in temp2:
        
        if word in idftable:
            den2 += (temp2[word]*idftable[word])*(temp2[word]*idftable[word])
        else:
            den2 += (temp2[word]*idftable[settings.OOV_WORD])*(temp2[word]*idftable[settings.OOV_WORD])
    
    
    for word in temp1:
        idf = idftable[settings.OOV_WORD]
        if word in idftable:
            idf = idftable[word]
        if word in temp2:
            num += (temp1[word]*idf)*(temp2[word]*idf)
            
    if den1==0 or den2==0:
        return 0
    else:
        return num/np.sqrt(den1*den2)


#####################################
    
    
discarddep={}

discarddep["det"]=""
discarddep["expl"]=""
discarddep["goeswith"]=""
discarddep["possessive"]=""
discarddep["preconj"]=""
discarddep["predet"]=""
discarddep["prep"]=""
discarddep["punct"]=""
discarddep["ref"]=""

def getSBAKSimilarity(sent1dpb, sent2dpb):
    
    sim = 0
#    print ("sent1 "+str(sent1dpb)+" "+str(len(sent1dpb)))
#    print ("sent2 "+str(sent2dpb)+" "+str(len(sent2dpb)))
    cnt = 0
    for bg1 in sent1dpb:
        (w11, t1, w12) = bg1
        
        if t1 in discarddep:
             continue
        
        for bg2 in sent2dpb:
            (w21, t2, w22) = bg2
            
            if t2 in discarddep:
                continue
        
            temp = 0 
            if w11.lower()==w21.lower():                
                temp += 1
            if w12.lower()==w22.lower():
                temp += 1
            if t1==t2:
                temp *= 1
            else:
                temp *= 0.5
            
           # print (str(bg1)+" "+str(bg2)+" "+str(temp))
#            print (w11==w21)
            cnt += 1 
            sim += temp
    
#    print (cnt)
#    print (str(len(sent1dpb)+len(sent2dpb)))
    if cnt==0:
        return 0
    else:
        return sim/(len(sent1dpb)+len(sent2dpb))
    
#####################################   
  
def createX(procsentmap, \
                postags, nertags,  \
                termmap, embedding_matrix, idfmap):
    
    simfeats = 3
    n_features = simfeats + len(postags) +len(nertags) 
    
   
    toreturn ={}
    sentnums=[]
    for sent1x in (procsentmap):
        sentnums.append(sent1x)
    
#    print (sentnums)
#    print (len(sentnums))
    for sx1, sent1x in enumerate(sentnums):
        
#        print ("here 1  "+str(sent1x))
        wds1, pos1, ner1, dbigs1 = procsentmap[sent1x]
        
        for sx2 in range(sx1+1, len(sentnums)):
           
            sent2x = sentnums[sx2]
#            print ("Here 2  "+str(sent2x))    
        
            x= np.zeros(n_features)
            wds2, pos2, ner2, dbigs2 = procsentmap[sent2x]
            
            lkupkey = str(sent1x)+" "+str(sent2x)
#            print ("pair "+lkupkey)
            if lkupkey not in toreturn:
                temp = str(sent2x)+" "+str(sent1x)
                if temp in toreturn:
                    continue
            
            x[0] = getEmbdgDotProduct(wds1, wds2, termmap, embedding_matrix)
            x[1] = getTFIDFCosine(wds1, wds2, termmap, idfmap)
            x[2] = getSBAKSimilarity(dbigs1, dbigs2)
            offset = simfeats
    
            for tagx, postag in enumerate(postags):
                x[offset + tagx] = \
                typeBasedSimilarity(postag, wds1, pos1, wds2, pos2)
        
            offset = offset + len(postags)
            for tagx, nertag in enumerate(nertags):
                x[offset + tagx] = \
                typeBasedSimilarity(nertag, wds1, ner1, wds2, ner2)
   
            toreturn[lkupkey] = x
            
    
    return toreturn