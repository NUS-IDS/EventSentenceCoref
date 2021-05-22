#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 23:15:43 2021

@author: sdas
"""

import pickle
import gloveLoader as gl
import ExptSettings as settings
import KMDataLoader as dl
import sys
import os
import km as km

def test(inpProcDir):
    
    inpfile = inpProcDir + "/simfeats.pkl"
    temp=pickle.load(open(inpfile, "rb"))
    
    for ix, iid in enumerate(temp):
        
        if ix < 3:
            
            for pair in temp[iid]:
                print (pair)
                print (temp[iid][pair])
        
    print (len(temp))
    
    return

def gendata(inpProcDir):
    
    outfile = inpProcDir + "/simfeats.pkl"
    
    iid2pairs = pickle.load(open(inpProcDir+"/iid2pairs.pkl", "rb"))
    
    iid2procsentmap = pickle.load(open(inpProcDir+"/iid2procsentmap.pkl", "rb"))
    
    print (len(iid2pairs))
    
   
     
    idfmap = pickle.load(open (inpProcDir+"/idfmap.pkl", "rb"))
    termmap = pickle.load(open (inpProcDir+"/termmap.pkl", "rb"))
    nertags = pickle.load(open (inpProcDir+"/nertags.pkl", "rb"))
    postags = pickle.load(open (inpProcDir+"/postags.pkl", "rb"))
    embedding_matrix = pickle.load(open (inpProcDir+"/embedding.pkl", "rb"))
    
    
    temp={}
    for ix, iid in enumerate(iid2pairs):
    
       
        procsentmap = iid2procsentmap[iid]
        
        
        featmap = km.createX(procsentmap, \
                    postags, nertags,  \
                     termmap, embedding_matrix, idfmap)
        
        print (iid+" "+str(len(featmap)))
        temp[iid] = featmap
        
    pickle.dump(temp, open(outfile, "wb"))
    
    
    temp={}
    for ix, iid in enumerate(iid2pairs):
    
        procsentmap = iid2procsentmap[iid]
        
        
        featmap = km.createX(procsentmap, \
                    postags, nertags,  \
                     termmap, embedding_matrix, idfmap)
        
        print (iid+" "+str(len(featmap)))
        temp[iid] = featmap
     
    pickle.dump(temp, open(outfile, "wb"))
    
    return

    
def getTermDicts(serializedDataDir, outputDir):
#serializedDataDir = "/home/sdas/fiona/processed"
    iid2pairs, iid2clustermap, \
    iid2sentmap, iid2procsentmap, iid2similarities = dl.loadSerializedData(serializedDataDir)
    
    postags, nertags = dl.getUniquePOSNERTags(iid2procsentmap)
    
    print ("len(postags)="+str(len(postags)))
    print ("len(nertags)="+str(len(nertags)))
    print ("len(iid2pairs)="+str(len(iid2pairs)))
    
   
    termdict, termmap, idfmap = dl.createTrainDictionary(settings.tthresh, \
                                iid2pairs, iid2procsentmap)
       
    print ("#terms in dictionary "+str(len(termdict)))    
    print ("OOVID "+str(termmap[settings.OOV_WORD]))
    
    embedding_matrix = \
        gl.load_glove_embedding(settings.weFile, termmap, settings.weDim)
    
    
    embdgf = outputDir +"/embedding.pkl"
    pickle.dump(embedding_matrix, open(embdgf, "wb"))
    print ("Dumped to "+embdgf)
    
    postagsf = outputDir +"/postags.pkl"
    pickle.dump (postags, open(postagsf, "wb"))
    print ("Dumped to "+postagsf)
    
    nertagsf = outputDir +"/nertags.pkl"
    pickle.dump (nertags, open(nertagsf, "wb"))
    print ("Dumped to "+nertagsf)
    
    termmapf = outputDir +"/termmap.pkl"
    pickle.dump (termmap, open(termmapf, "wb"))
    print ("Dumped to "+termmapf)
    
    idfmapf = outputDir +"/idfmap.pkl"
    pickle.dump (idfmap, open(idfmapf, "wb"))
    print ("Dumped to "+idfmapf)


if __name__ == '__main__':
    
    if len(sys.argv)!=3:
        print ("\n\nargs1: en-inpdir-json-files, args2: outdir")
        
    else:
        inpdir = sys.argv[1]
        procdir = sys.argv[2]
        inpfileslist=[]
        
        for file in os.listdir(inpdir):
            
            if file.endswith(".json"):
                inpfileslist.append(inpdir+"/"+file)
                
        print ("Processing files")
        print (inpfileslist)
        print ("Output will be written to "+procdir)
        
        if len(inpfileslist) > 0:
            dl.serializeData(inpfileslist, procdir)
            getTermDicts(procdir, procdir)
            gendata(procdir)
            test(procdir)