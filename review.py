"""
this is a replica of the 'review' function in 'main.py' 
this script just loads fewer dependencies for a faster run when in reviewing mode
(i.e. you have already trained the model and saved the predictions in csv)
"""
# packages
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from collections import defaultdict
import os
import re
import time
import argparse
import sys
import pickle
import pandas as pd

# project
from src.utils.files import open_json, save_json, make_dir, set_seeds, set_warnings, get_device, str2bool
from src.utils.logger import get_logger, log_params, get_log_level, save_results_to_csv
from src.process.trainer import convert_to_sklearn_format, calc_doc_score, pairs_to_doc, get_additional_scores

# args
parser = argparse.ArgumentParser()
parser.add_argument('--run_type', type=str, default='run_kfolds',
                    help='process to run | options: run_one, run_kfolds, train, evaluate')
parser.add_argument('--model_name', type=str, default='bilstm', 
                    help='name of the model | options: bilstm, linear, kmeans, lda, dbscan')
parser.add_argument('--bert_model', type=str,
                    default='bert-base-cased', help='name of pretrained BERT model')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size for train and evaluate')
parser.add_argument('--folds', type=int, default=5,
                    help='number of training folds')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of training epochs')
parser.add_argument('--best_epoch','--list', nargs='+', default=[],
                    help='Overwrites best epoch per fold to use in pairs2doc, must be in a list format (E.g. -best_epoch 9 8 4 3 8)')
parser.add_argument('--val_size', type=float, default=0.2,
                    help='proportion of data to use for validation')
parser.add_argument('--optimizer', type=str,
                    default='adam', help='name of optimizer | options: adam, sgd')
parser.add_argument('--lr_rate', type=float, default=2e-5,
                    help='learning rate of optimizer')
parser.add_argument('--dropout_rate', type=float, default=0.3,
                    help='dropout rate between layers')
parser.add_argument('--max_len', type=int, help='maximum length of sentences')
parser.add_argument('--lstm_out_dim', type=int, default=64,
                    help='dimension size for bilstm output, int>=2')
parser.add_argument('--use_cw', type=str2bool, nargs='?', default=True,
                    help='calculate class weights from data and include in loss calculations')
parser.add_argument('--use_pos', type=str2bool, nargs='?', default=True,
                    help='use pos information in pipeline. must be bilstm.')
parser.add_argument('--pos_type', type=str, default='upos',
                    help='stanza upos or xpos, note: multilingual only has upos support')
parser.add_argument('--use_info', type=str2bool, nargs='?', default=True,
                    help='use pair info in pipeline. must be linear or bilstm')
parser.add_argument('--use_suj', type=str2bool, nargs='?', default=False,
                    help='use additional pair info in pipeline. must be linear or bilstm')
parser.add_argument('--early_stopping', type=str2bool, nargs='?', default=True,
                    help='use last epoch or best epoch based on validation set')
parser.add_argument('--mask', type=str2bool, nargs='?', default=False,
                    help='use additional pair info in pipeline. must be linear or bilstm')
parser.add_argument('--strip_title', type=str2bool, nargs='?', default=True,
                    help='remove title from first sentence')
parser.add_argument('--repeat', type=str2bool, nargs='?', default=False,
                    help='flips head tails to get double number of pairs to train on (classify_by must be pairs or mix)')
parser.add_argument('--double_train', type=str2bool, nargs='?', default=False,
                    help='do additional training by doc. must be linear or bilstm')

# document level
parser.add_argument('--classify_by', type=str, default='pairs',
                    help='specify to do classification task by sentence pairs or document level | options: pairs')
parser.add_argument('--clus_method', type=str, default='ors',
                    help='pairs to docs clustering method | options: graph, hc, ors')

# general
parser.add_argument('--data_folder', type=str, default='data',
                    help='folder name where data is located')
parser.add_argument('--train_data_name', type=str,
                    default='train.json', help='name of train data file')
parser.add_argument('--test_data_name', type=str,
                    default='test.json', help='name of test data file')
parser.add_argument('--out_folder', type=str, default='outs',
                    help='folder name to save outputs into')
parser.add_argument('--log_file', type=str,
                    default='training.log', help='filename to save log')
parser.add_argument('--results_file', type=str,
                    default='results.csv', help='filename to save results summary')
parser.add_argument('--plot_save_name', type=str, default='training_results_plot.png',
                    help='filename to save results plot in png format')
parser.add_argument('--model_save_name', type=str, default='best_model_state.bin',
                    help='filename to save best model in bin format')
parser.add_argument('--train_save_name', type=str,
                    help='filename to save training results in csv format (e.g. train_res.csv)')
parser.add_argument('--val_save_name', type=str, default='val_res.csv',
                    help='filename to save validation results in csv format (e.g. val_res.csv)')
parser.add_argument('--test_save_name', type=str, default='test_res.csv',
                    help='filename to save unseen test results in csv format (e.g. test_res.csv)')
parser.add_argument('--use_backup', type=str2bool, nargs='?',
                    default=True, help='load nlp parsed items if already available')
parser.add_argument('--translate_from', type=str,
                    help='language to translate from (e.g. es [Spanish], pt [Portugese]), default None')

parser.add_argument('--predict', type=str2bool, nargs='?', default=False,
                    help='to predict upon unseen test set or not (False to save computation when experimenting)')
parser.add_argument('--build_mode', type=str2bool, nargs='?',
                    default=False, help='activate build mode (aka reduced data size)')
parser.add_argument('--use_cpu', type=str2bool, nargs='?', default=False,
                    help='overwrite and use cpu (even if gpu is available)')
parser.add_argument('--cuda_device', type=str, default='0',
                    help='set which cuda device to run on (0 or 1)')
parser.add_argument('--log_level', type=str, default='info',
                    help='set logging level to store and print statements')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed to set for reproducibility')

args = parser.parse_args()

# setting up
set_seeds(args.seed)
set_warnings()
log_save_path = f'{args.out_folder}/{args.model_name}/{args.log_file}'.lower()
make_dir(log_save_path)
logger = get_logger(log_save_path, no_stdout=False, set_level=args.log_level)
device = get_device(args.use_cpu, args.cuda_device)

# suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# functions
def trainer_pairs2doc(best_epoch=None, folds_list=None):
    logger.info('-- converting pairs2doc')

    gold_df = open_json(os.path.join(args.data_folder, args.train_data_name), data_format=pd.DataFrame)

    if folds_list is None:
        if args.folds==0:
            folds_list = ['']
        else:
            folds_list = [f'K{fold}_' for fold in range(args.folds)]
        foldt = ''
    else:
        foldt = ''.join(folds_list)

    if len(args.best_epoch)>0:
        logger.info('-- overwriting best epoch')
        if args.run_type == 'review' and len(folds_list)==1:
            fold_num = int(re.findall(r'\d+', folds_list[0])[0])
            best_epoch = args.best_epoch[fold_num]
        else:
            best_epoch = args.best_epoch

    if best_epoch is None:
        best_epoch = args.epochs-1
    # logger.info(f'Evaluating {folds_list} datasets...')

    logger.debug(f'folds_list: {folds_list} | best_epoch: {best_epoch}')

    if args.train_save_name is not None:
        pred_train = pairs_to_doc(
            args.out_folder, args.model_name, folds_list, best_epoch, args.train_save_name, args.clus_method)

        train_save_path = None if args.train_save_name is None \
            else f'{args.out_folder}/{args.model_name}/{foldt}_doc_{args.clus_method}_{args.train_save_name}'

        calc_doc_score(
            gold_df.loc[pred_train.keys()],
            pred_train.values(), 
            foldt=f'Train_{foldt}',
            save_data_path=train_save_path,
            keys=list(pred_train.keys())
            )
        get_additional_scores(
            goldfile=os.path.join(args.data_folder, args.train_data_name), 
            sysfile=train_save_path, foldt=f'Train_{foldt}',
            )

    if args.val_size>0:
        pred_val = pairs_to_doc(
            args.out_folder, args.model_name, folds_list, best_epoch, args.val_save_name, args.clus_method)
        val_save_path = f'{args.out_folder}/{args.model_name}/{foldt}_doc_{args.clus_method}_{args.val_save_name}'
        calc_doc_score(
            gold_df.loc[pred_val.keys()],
            pred_val.values(), 
            foldt=f'Val_{foldt}',
            save_data_path=val_save_path,
            keys=list(pred_val.keys())
            )
        get_additional_scores(
            goldfile=os.path.join(args.data_folder, args.train_data_name), 
            sysfile=val_save_path, foldt=f'Val_{foldt}'
            )

    if args.predict:
        pred_test = pairs_to_doc(
            out_folder=args.out_folder, model_name=args.model_name, folds_list=folds_list, 
            best_epoch=best_epoch, val_save_name=args.test_save_name, clustering_method=args.clus_method,
            val_file_path=None, mode='test')
        calc_doc_score(
            None,
            pred_test.values(), 
            foldt=f'Test_{foldt}',
            save_data_path=f'{args.out_folder}/{args.model_name}/{foldt}_doc_{args.clus_method}_{args.test_save_name}',
            keys=list(pred_test.keys())
            )


def review_validation():
    "for reviewing KFold validation scores"
    for fold in range(args.folds):
        trainer_pairs2doc(folds_list=[f'K{fold}_'])
    trainer_pairs2doc()


def main(task):
    task_func = {
        'review': review_validation
    }
    task_func[task]()


if __name__ == "__main__":
    logger.info('-- starting process')
    log_params(args)
    main('review')
    save_results_to_csv(os.path.join(args.out_folder, args.results_file))
    logger.info('-- complete process')
