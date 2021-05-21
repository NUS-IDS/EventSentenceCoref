import re
import logging
from google_trans_new import google_translator


translator = google_translator()

dt_formats = {
    'archive.indianexpress':'(\w{3}\s+\w{3}\s+\d{2}\s+\d{4}.+\d{2}:\d{2}\shrs\s)|(IST.+,\s\w+\s\d+\s:)',
    'newindianexpress': '((\w+\s\d*\s,\s\d{4}\s:)|(\d{4}\s+\d{2}:\d{2}\s\w{2}\s(.+:)*))',
    'thehindu': '(\w+\s\d{2}\s,\s\d{4}\s\d{2}:\d{2}\sIST\s((\S+\s){0,5}:\s)?)',
    'scmp': '(PUBLISHED\s+:.+\d{2}\s+\w+.+\d{4}.+:\d{2}\w{2}\s)'
}


def clean_data(list_df, strip_title, translate_from, doc_tag=None):
    """
    Mass clean document
    Implemented: Strips title and time in first sentence for AESPEN dataset
    """
    if translate_from is None and doc_tag is None and not strip_title:
        # no edits
        return list_df
    else:
        keep = []
        for ddict in list_df:
            doc = ddict['sentences']
            if translate_from is not None:
                doc = translate_doc(doc, translate_from)
            if strip_title and 'url' in ddict.keys():
                url = ddict['url']
                doc = clean_doc(doc, url)
            ddict['sentences'] = doc
            if doc_tag is not None:
                ddict['tag'] = doc_tag
            keep.append(ddict)
        return keep


def clean_doc(doc, url):
    """
    Manual functions to clean document
    """
    
    ##### Remove titles/datetime in first sentence #####
    default_pattern = '(^((\S+\s){1,3}:))'
    dt_pattern = [dt_formats[k] for k in dt_formats.keys() if k in url]
    if len(dt_pattern) > 0:
        dt_pattern = dt_pattern[0]+'|'+default_pattern
    else:
        # no title/datetime format found, use default
        dt_pattern = default_pattern
    
    splits = list(filter(lambda x:x!=None, re.split(dt_pattern, doc[0])))
    if len(splits)>1:
        doc[0] = splits[-1].lstrip()
    else:
        # no title/datetime format found
        pass
    
    return doc


def translate_doc(doc, translate_from):
    keep = []
    for sent in doc:
        keep.append(translator.translate(sent, lang_src=translate_from, lang_tgt='en'))
    return keep



