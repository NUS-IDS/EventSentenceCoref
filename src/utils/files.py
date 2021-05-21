import json
import os
import pandas as pd
import numpy as np
import warnings
import argparse
import torch
import random
from pathlib import Path
from nltk.corpus import stopwords


def open_json(json_file_path, data_format=list):
    if data_format==dict or data_format=='dict':
        with open(json_file_path) as json_file:
            data = json.load(json_file)
    elif data_format==list or data_format=='list':
        data = []
        for line in open(json_file_path, encoding='utf-8'):
            data.append(json.loads(line))
    elif data_format==pd.DataFrame or data_format=='pd.DataFrame':
        data = pd.read_json(json_file_path, orient="records", lines=True)
    else:
        raise NotImplementedError
    return data


def save_json(ddict, json_file_path):
    with open(json_file_path, 'w') as fp:
        json.dump(ddict, fp)


def make_dir(save_path):
    path = Path(save_path)
    if not os.path.isdir(path.parent):
        os.makedirs(path.parent, exist_ok=True)


def set_seeds(seed):
    # for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def set_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)


def str2bool(v):
    """
    Code source: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_device(use_cpu, cuda_device):
    if torch.cuda.is_available() and not use_cpu:
        device = torch.device('cuda:'+str(cuda_device))
    else:
        device = torch.device("cpu")
    return device


def get_stop_words(language='english'):
    my_stop_words = stopwords.words(language)
    my_stop_words.extend([
        'could', 'might', 'must', 'need', 'shall', 'would', 
        "'d", "'ll", "'re", "'s", "'ve", "n't", 'sha', 'wo']
        )
    return my_stop_words


# def get_device(local_rank, use_cuda):
#     """
#     Get torch device and number of gpus.

#     Parameters
#     ----------
#     local_rank : int
#         local_rank for distributed training on gpus
#     use_cuda : bool
#         use cuda if available
#     """
#     if local_rank == -1 or not use_cuda:
#         device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
#         n_gpu = torch.cuda.device_count()
#     else:
#         torch.cuda.set_device(local_rank)
#         device = torch.device("cuda", local_rank)
#         n_gpu = 1
#         # Initializes the distributed backend which will
#         # take care of sychronizing nodes/GPUs
#         torch.distributed.init_process_group(backend='nccl')

#     return device, n_gpu
