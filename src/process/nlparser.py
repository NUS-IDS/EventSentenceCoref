import re
import math
import pickle
import logging
import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import stanza
from gensim.models import Word2Vec
import gensim.downloader as api
from ..utils.files import get_stop_words
# from allennlp.predictors.predictor import Predictor


##### pretrained models #####
# stanza.download('es')
en_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,ner,lemma')
es_nlp = stanza.Pipeline(lang='es', processors='tokenize,pos,ner,lemma')
pt_nlp = stanza.Pipeline(lang='pt', processors='tokenize,pos,lemma')
glove_model = api.load("glove-wiki-gigaword-50")
custom_stopwords = get_stop_words()
# model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
# coref_predictor = Predictor.from_path(model_url)


##### parse texts #####
def safe_retrieve(pair, contents, url):
    suj_infos = [0]*27
    key = str(pair[0])+' '+str(pair[1])
    if url in contents.keys():
        if key in contents[url].keys():
            suj_infos = list(contents[url][key])
    return suj_infos


def format_into_pairs(input_df, return_start_ix=False, fuzz_threshold=90, token_to_ix=None, \
    mode='train', repeat=False, multilingual=False, random_masking=False, pos_type='upos', use_suj=False):

    """
    input dict: keys --> ['url', 'sentences', 'sentence_no', 'event_clusters']
    output dict: keys --> ['pair', 'target', 'source']
    reference dict: eg_ix --> 'sentence_no':  
        sentences : 'sentences'
        postags : 'pos'
    """
    logging.info(f'Creating using Multilingual? {multilingual}')

    if use_suj:
        with open("data/simfeats.pkl", "rb") as f:
            contents = pickle.load(f)
        with open("data/en-simfeats.pkl", "rb") as f:
            en_contents = pickle.load(f)
        with open("data/es-simfeats.pkl", "rb") as f:
            es_contents = pickle.load(f)
        with open("data/pt-simfeats.pkl", "rb") as f:
            pt_contents = pickle.load(f)

    if token_to_ix is None:
        # initialise
        token_to_ix = {'[MASK]': 0, '[CLS]':1, '[SEP]':2}

    data, reference = {}, {}
    doc_start_ix = []
    counter = 0

    for eg_ix in input_df.keys():
        # create doc info
        language = input_df[eg_ix]['tag']
        doc_start_ix.append(counter)
        doc_dict = {}
        # sentences = coref_resolve_doc(input_df[eg_ix]['sentences'])
        for ix, sent_no in enumerate(input_df[eg_ix]['sentence_no']):
            # resolved_sentence = sentences[ix]
            sentence = input_df[eg_ix]['sentences'][ix].strip()
            if multilingual:
                enc_sent = get_encoded_sentence(sentence, language)
            else:
                enc_sent = en_nlp(sentence)
            if pos_type == 'upos':
                pos = [word.upos for sent in enc_sent.sentences for word in sent.words]
                noun = 'NOUN'
                verb = 'VERB'
            else:
                pos = [word.xpos for sent in enc_sent.sentences for word in sent.words]
                noun = 'NN'
                verb = 'VB'
            lemma = [word.lemma for sent in enc_sent.sentences for word in sent.words]
            key_nouns = [s for s, p in zip(lemma, pos) if noun in p]
            key_verbs = [s for s, p in zip(lemma, pos) if verb in p]
            key_verbs = [s for s in key_verbs if s not in custom_stopwords]
            if multilingual and language == 'pt':
                # stanza ner not available for pt at the moment
                key_entities = {}
            else:
                key_entities = {ent.text: ent.type for sent in enc_sent.sentences for ent in sent.ents}
            for i in pos:
                if i not in token_to_ix.keys():
                    token_to_ix[i]=max(token_to_ix.values())+1
            if random_masking:
                sentence = ' '.join([np.random.choice([word, '[MASK]'], p=[0.95, 0.05]) for word in sentence.split(' ')])
            doc_dict[sent_no] = {
                'sentence': sentence,
                'lemma': lemma,
                'pos': np.stack([create_one_hot(1)]+[create_one_hot(token_to_ix[i]) for i in pos], axis=0),
                'key_nouns': key_nouns,
                'key_verbs': key_verbs,
                'entities': key_entities#,
                # 'entities2': list(set([i for i in resolved_sentence if i if re.match("ent\d\d", i)]))
            }
        reference[eg_ix] = doc_dict

        # create pairs info
        combis = list(combinations(input_df[eg_ix]['sentence_no'], 2))
        ddict = dict.fromkeys(combis, 0)
        info = {}
        for pair in ddict.keys():
            i_list = list(reference[eg_ix][pair[0]]['entities'].keys())
            j_list = list(reference[eg_ix][pair[1]]['entities'].keys())
            ent_count = [fuzz.token_sort_ratio(i,j) for i in i_list for j in j_list]
            ent_count = len([i for i in ent_count if i>fuzz_threshold])
            ent_count_wt = safe_div(ent_count, len(set(i_list+j_list)))
            # i_list = reference[eg_ix][pair[0]]['entities2']
            # j_list = reference[eg_ix][pair[1]]['entities2']
            # ent_count2 = sum([1 for i in i_list if i in j_list])
            # ent_count2_wt = safe_div(ent_count2, len(set(i_list+j_list)))

            i_list = reference[eg_ix][pair[0]]['key_nouns']
            j_list = reference[eg_ix][pair[1]]['key_nouns']
            noun_sim = [1 for i in i_list for j in j_list if i.lower()==j.lower()]
            noun_count = sum(noun_sim) if len(noun_sim)>0 else 0
            noun_count_wt = safe_div(noun_count, len(set(i_list+j_list)))
            # noun_sim = [glove_model.similarity(i.lower(),j.lower()) for i in i_list for j in j_list
            #             if i.lower() in glove_model.vocab.keys() and j.lower() in glove_model.vocab.keys()]
            # max_noun = max(noun_sim) if len(noun_sim)>0 else 0
            # mean_noun = sum(noun_sim)/len(noun_sim) if len(noun_sim)>0 else 0

            i_list = reference[eg_ix][pair[0]]['key_verbs']
            j_list = reference[eg_ix][pair[1]]['key_verbs']
            verb_sim = [1 for i in i_list for j in j_list if i.lower()==j.lower()]
            verb_count = sum(verb_sim) if len(verb_sim)>0 else 0
            verb_count_wt = safe_div(verb_count, len(set(i_list+j_list)))
            # verb_sim = [glove_model.similarity(i.lower(),j.lower()) for i in i_list for j in j_list
            #             if i.lower() in glove_model.vocab.keys() and j.lower() in glove_model.vocab.keys()]
            # max_verb = max(verb_sim) if len(verb_sim)>0 else 0
            # mean_verb = sum(verb_sim)/len(verb_sim) if len(verb_sim)>0 else 0

            infos = [
                verb_count, noun_count, ent_count,
                verb_count_wt, noun_count_wt, ent_count_wt
                #ent_count2, ent_count2_wt
                ]
            if use_suj:
                sl = 27
                suj_infos = [0]*sl
                if 'url' in input_df[eg_ix].keys():
                    # for AESPEN format
                    suj_infos = safe_retrieve(pair, contents, input_df[eg_ix]['url'])
                elif 'id' in input_df[eg_ix].keys():
                    # for CASE format
                    if language=='en':
                        suj_infos = safe_retrieve(pair, en_contents, str(input_df[eg_ix]['id']))
                    elif language=='es':
                        suj_infos = safe_retrieve(pair, es_contents, str(input_df[eg_ix]['id']))
                    elif language=='pt':
                        suj_infos = safe_retrieve(pair, pt_contents, str(input_df[eg_ix]['id']))
                length = len(suj_infos)
                suj_infos += [0]*(sl-length)
                infos += suj_infos[0:sl]
            info[pair] = infos

        # update same cluster event
        if mode=='train':
            for clus in input_df[eg_ix]['event_clusters']:
                clus_combis = list(combinations(clus, 2))
                if len(clus_combis)==0:
                    next
                for pair in clus_combis:
                    ddict[pair]=1

        for key, value in ddict.items():
            data[counter] = {'pair':key, 'target':value, 'source':eg_ix, 'info': info[key]}
            counter+=1
            if repeat and mode=='train':
                h, t = key
                data[counter] = {'pair':(t,h), 'target':value, 'source':eg_ix, 'info': info[key]}
                counter+=1
        
        logging.debug(data[counter-1])

    # append last number
    doc_start_ix.append(counter)

    if return_start_ix:
        return data, reference, token_to_ix, doc_start_ix
    else:
        return data, reference, token_to_ix


def get_encoded_sentence(sentence, language):
    if language == 'pt' or language == 'ps':
        return pt_nlp(sentence)
    elif language == 'es':
        return es_nlp(sentence)
    else:
        return en_nlp(sentence)


def create_one_hot(pos, max_pos=60):
    l = np.zeros(max_pos)
    l[pos] = 1
    return l


def safe_div(x, y):
    if y == 0:
        return 0
    return x / y


def coref_resolve(t, clusters):
    clus_no = 0
    for clus in clusters:
        for ent in clus:
            for i in range(ent[0],ent[1]+1):
                if i==0:
                    t = ["ent{:02d}".format(clus_no)] + t[i+1:]
                elif i==len(t):
                    t = t[:i] + ["ent{:02d}".format(clus_no)]
                else:
                    t = t[:i] + ["ent{:02d}".format(clus_no)] + t[i+1:]
        clus_no += 1
    return t


def coref_resolve_doc(list_of_sentences):
    sentences, indices = [], []
    total_length = 0
    for i in list_of_sentences:
        tok_sent = i.strip().split(' ')
        start = total_length
        total_length += len(tok_sent)
        end = total_length-1
        sentences.extend(tok_sent)
        indices.append((start,end))
    logging.debug(sentences)
    p = coref_predictor.predict_tokenized(sentences)
    sentences = coref_resolve(p['document'], p['clusters'])
    return [sentences[s:e+1] for s,e in indices]

