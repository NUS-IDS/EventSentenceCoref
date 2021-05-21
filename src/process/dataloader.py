import torch
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .nlparser import create_one_hot

class EventNewsDataset(Dataset):
    """
    Generate Pytorch dataset from EventNewsDataset in pairs format
    """
    
    def __init__(self, keys, data, reference, tokenizer, max_len):
        self.keys = keys
        self.data = data
        self.reference = reference
        self.tokenizer = tokenizer
        self.max_len = max_len
  
    def __len__(self):
        return len(self.keys)

    def custom_tokenizer(self, text, text_pair=None):
        return self.tokenizer.encode_plus(
            text=text,
            text_pair=text_pair,
            add_special_tokens=True,
            max_length=1+(self.max_len+1)*2,
            truncation=True,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            ) 
            # using bert-base wordpiece tokenizer
            # e.g. https://huggingface.co/transformers/v2.4.0/glossary.html
  
    def __getitem__(self, item):
        key = self.keys[item]
        pairsdict = self.data[key]
        head, tail = pairsdict['pair']
        target = pairsdict['target']
        source = pairsdict['source']
        info = pairsdict['info']
        eg_ids = f'{source}: ({head},{tail})'

        # add pos info
        m = 1+(self.max_len+1)*2
        _head = self.reference[source][head]['pos']
        _tail = self.reference[source][tail]['pos']
        max_pos = 60
        pad_len = m-(len(_head)+len(_tail))-1
        pair_pos = np.concatenate([
            _head, _tail,
            np.expand_dims(create_one_hot(2), axis=0),
            np.broadcast_to(create_one_hot(1), (pad_len,)+(max_pos,))
            ], axis=0)
        
        # add lemma info
        _head = self.reference[source][head]['lemma']
        _tail = self.reference[source][tail]['lemma']
        pair_lemma = self.custom_tokenizer(_head, _tail)['input_ids']
        logging.debug(f'pair_pos.shape: {pair_pos.shape}')
        logging.debug(f'pair_lemma.shape: {pair_lemma.shape}')
        pair_pos = np.concatenate([pair_pos, pair_lemma.permute(1,0)], axis=1)
        logging.debug(f'pair_pos.shape: {pair_pos.shape}')
        
        _head = str(self.reference[source][head]['sentence'])
        _tail = str(self.reference[source][tail]['sentence'])
        enc = self.custom_tokenizer(_head, _tail)

        return {
            'pairs': (_head, _tail), # sentence text, not used in the end (reevaluate to drop)
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'token_type_ids': enc['token_type_ids'].squeeze(),
            'pair_pos': torch.tensor(pair_pos, dtype=torch.float64),
            'pair_infos': torch.tensor(info, dtype=torch.float64),
            'targets': torch.tensor(target, dtype=torch.long),
            'eg_ids': eg_ids
        }

        # head_enc = self.custom_tokenizer(head)
        # tail_enc = self.custom_tokenizer(tail)

        # return {
        #     'pairs': (head, tail), # sentence text, not used in the end (reevaluate to drop)
        #     'input_ids': torch.cat(
        #         [head_enc['input_ids'].flatten(), tail_enc['input_ids'].flatten()], dim=0),
        #     'attention_mask': torch.cat(
        #         [head_enc['attention_mask'].flatten(), tail_enc['attention_mask'].flatten()], dim=0),
        #     'targets': torch.tensor(target, dtype=torch.long),
        #     'eg_ids': eg_ids
        # }

class PairsByDocDataset(object):
    """
    Generate custom dataset from PairsByDocDataset in pairs format per document
    """
    
    def __init__(self, start_ix, data, doc_clus, reference, tokenizer, max_len, drop_last):
        self.start_ix = start_ix
        self.data = data
        self.doc_clus = doc_clus
        self.reference = reference
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.drop_last = drop_last

    def __len__(self):
        return len(self.doc_clus)

    def __getitem__(self, item):
        pair_ixes = list(range(self.start_ix[item],self.start_ix[item+1]))
        source = self.data[pair_ixes[0]]['source']
        logging.debug(f'item {item}, eg_ix {source}, pair_ixes {pair_ixes}')

        ds = EventNewsDataset(
            keys=np.array(pair_ixes),
            data=self.data,
            reference=self.reference,
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )

        dl = DataLoader(
            ds,
            drop_last=self.drop_last,
            batch_size=16, # FIXED FOR NOW
            num_workers=4,
            shuffle=True
        )

        return {
            'dl': dl,
            'doc_clus': torch.tensor(self.doc_clus[source], dtype=torch.long),
            'source': source
        }


class PairsnDocDataset(Dataset):
    """
    Generate Pytorch dataset from PairsnDocDataset in pairs format per document
    """
    
    def __init__(self, start_ix, pairs, targets, sources, doc_clus, reference, tokenizer, max_len):
        self.start_ix = start_ix
        self.pairs = pairs
        self.targets = targets
        self.sources = sources
        self.doc_clus = doc_clus
        self.reference = reference
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.doc_clus)

    def custom_tokenizer(self, text, text_pair=None):
        return self.tokenizer.encode_plus(
            text=text,
            text_pair=text_pair,
            add_special_tokens=True,
            max_length=1+(self.max_len+1)*2,
            truncation=True,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            ) 

    def __getitem__(self, item):
        pair_ixes = list(range(self.start_ix[item],self.start_ix[item+1]))
        source = self.sources[pair_ixes[0]]
        logging.debug(f'item {item}, eg_ix {source}, pair_ixes {pair_ixes}')

        head, tail, eg_ids = [], [], []
        input_ids, attn_mask = [], []
        for p in self.pairs[pair_ixes]:
            h, t = p[0], p[1]
            head.append(h)
            tail.append(t)
            enc = self.custom_tokenizer(
                self.reference[source][h]['sentence'], 
                self.reference[source][t]['sentence'])
            input_ids.append(enc['input_ids'])
            attn_mask.append(enc['attention_mask'])
            eg_ids.append(f'{source}: ({h},{t})')

        logging.debug(f'len of input_ids: {len(input_ids)}')
        logging.debug(f'len of attn_mask: {len(attn_mask)}')
        # len can be 78 here cause it's number of pairs

        return {
            'source': source,
            'input_ids': torch.stack(input_ids, dim=0).view(len(pair_ixes),-1),
            'attention_mask': torch.stack(attn_mask, dim=0).view(len(pair_ixes),-1),
            'targets': torch.tensor(self.targets[pair_ixes], dtype=torch.long),
            'doc_clus': torch.tensor(self.doc_clus[source], dtype=torch.long),
            'eg_ids': eg_ids
        }


class SentenceDataset(Dataset):
    def __init__(self, sents, tokenizer, max_len):
        self.sents = sents
        self.tokenizer = tokenizer
        self.max_len = max_len
  
    def __len__(self):
        return len(self.sents)

    def custom_tokenizer(self, text):
        return self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            )

    def __getitem__(self, item):
        sent = self.sents[item]
        sent_enc = self.custom_tokenizer(sent)
        return {
            'input_ids': sent_enc['input_ids'].flatten(),
            'attention_mask': sent_enc['attention_mask'].flatten()
        }


def create_data_loader(data, reference, tokenizer, max_len, batch_size, drop_last, sent_format='pairs'):
    if sent_format == 'pairs':
        ds = EventNewsDataset(
            keys=np.array(list(data.keys())),
            data=data,
            reference=reference,
            tokenizer=tokenizer,
            max_len=max_len
        )
    elif sent_format == 'ddl':
        ds = PairsByDocDataset(
            start_ix=data['start_ix'],
            data=data['pairs'],
            doc_clus=data['doc_clus'],
            reference=reference,
            tokenizer=tokenizer,
            max_len=max_len,
            drop_last=drop_last
        )
        return ds
    elif sent_format == 'one':
        ds = PairsnDocDataset(
            start_ix=data['start_ix'],
            pairs=np.array([v['pair'] for k,v in data['pairs'].items()]),
            targets=np.array([v['target'] for k,v in data['pairs'].items()]),
            sources=np.array([v['source'] for k,v in data['pairs'].items()]),
            doc_clus=data['doc_clus'],
            reference=reference,
            tokenizer=tokenizer,
            max_len=max_len
        )
    else:
        ds = SentenceDataset(
            sents=data,
            tokenizer=tokenizer,
            max_len=max_len
        )

    return DataLoader(
        ds,
        drop_last=drop_last,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )