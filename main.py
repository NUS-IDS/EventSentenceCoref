# packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
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
from torch import optim
from transformers import BertTokenizer, AlbertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import adjusted_rand_score
from kneed import KneeLocator

# project
from src.utils.files import open_json, save_json, make_dir, set_seeds, set_warnings, get_device, \
    str2bool, get_stop_words
from src.utils.logger import get_logger, log_params, get_log_level, save_results_to_csv
from src.process.nlparser import format_into_pairs
from src.process.cleaner import clean_data
from src.process.formatter import get_k_train_test_data, get_train_test_data, get_max_sent_len, \
    get_class_weight, get_max_sent_k, get_x_y_data, compare_pred_vs_actual, force_extend_label_levels, \
        get_max_doc_len
from src.process.dataloader import create_data_loader
from src.process.trainer import train_step, evaluate_step, pairs_to_doc, get_additional_scores, calc_doc_score
from src.models.bilstm import BiLSTMPairClassifier

# args
# to do: option to evaluate only (load trained model and evaluate)

# pairs level
model_options = {
    'tok': [
        'bert-base-cased', 'bert-base-uncased', 'albert-base-v2', 'albert-xlarge-v2', 
        'tfidf', 'doc2vec', 'lda'
        ],
    'sent': ['bilstm', 'linear'],
    'doc': ['kmeans', 'lda', 'dbscan']
}

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
                    help='pairs to docs clustering method | options: graph, hc, ors, ors_rsc, ors_orig')

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
def plot(results, plot_save_path):
    if 'train_acc' in results.keys() and 'val_acc' in results.keys():
        plt.plot(results['train_acc'], label='train accuracy')
        plt.plot(results['val_acc'], label='validation accuracy')

        plt.title('Training results')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.ylim([0, 1])
        plt.savefig(plot_save_path, bbox_inches='tight')
        plt.close()
        logger.info('-- plotted chart')


def predict_pairs(foldt=''):
    multilingual = True if 'multilingual' in args.bert_model else False
    model_params = open_json(f'{args.out_folder}/{args.model_name}/{foldt}{args.model_save_name}.json', dict)
    max_len = int(((int(model_params['seq_len'])-1)/2)-1)
    
    test_df = get_unseen()
    test_df, test_ref, token_to_ix = format_into_pairs(test_df, mode='test', multilingual=multilingual,
    use_suj=args.use_suj)

    tokenizer = get_tokenizer()
    test_dl = create_data_loader(
        data=test_df,
        reference=test_ref,
        tokenizer=tokenizer,
        max_len=max_len,
        batch_size=args.batch_size,
        drop_last=False
    )
    del(test_df)
    logger.info('-- formatted files')

    model = BiLSTMPairClassifier(device=device, **model_params)
    model.load_state_dict(torch.load(f'{args.out_folder}/{args.model_name}/{foldt}{args.model_save_name}'))
    model.eval()
    model = model.to(device)
    logger.info('-- loaded model')

    criterion = nn.CrossEntropyLoss().to(device)
    _ = evaluate_step(
        model,
        test_dl,
        criterion,
        device,
        len(test_dl.dataset),
        f'{args.out_folder}/{args.model_name}/{foldt}pred_{args.test_save_name}'
    )
    pred_test = pairs_to_doc(
        val_file_path=f'{args.out_folder}/{args.model_name}/{foldt}pred_{args.test_save_name}', 
        clustering_method=args.clus_method
        )
    calc_doc_score(
        None,
        pred_test.values(), 
        foldt=f'Pred_',
        save_data_path=f'{args.out_folder}/{args.model_name}/{foldt}pred_doc_{args.test_save_name}',
        keys=list(pred_test.keys())
    )


def study_model(foldt=''):
    multilingual = True if 'multilingual' in args.bert_model else False
    model_params = open_json(f'{args.out_folder}/{args.model_name}/{foldt}{args.model_save_name}.json', dict)
    max_len = int(((int(model_params['seq_len'])-1)/2)-1)
    model = BiLSTMPairClassifier(device=device, **model_params)
    model.load_state_dict(torch.load(f'{args.out_folder}/{args.model_name}/{foldt}{args.model_save_name}'))
    model.eval()
    model = model.to(device)
    logger.info('-- loaded model')

    linearlayer = model.out.weight.cpu().detach().numpy()
    res_summary = pd.DataFrame(linearlayer)
    res_summary['labels'] = [0,1]
    res_summary['model_name'] = args.model_name

    save_file_path = 'outs/model_weights.csv'
    if os.path.isfile(save_file_path):
        # load old file
        old_summary = pd.read_csv(save_file_path)
        # append below
        res_summary = pd.concat([old_summary, res_summary], axis=0)
    res_summary.to_csv(save_file_path, index=False)


def get_model_params(model):
    params = {}
    params['seq_len'] = model.seq_len
    params['n_classes'] = model.n_classes
    params['lstm_out_dim'] = model.lstm_out_dim
    params['bert_model'] = model.bert_model
    params['dropout_rate'] = model.dropout_rate
    params['use_bilstm'] = model.use_bilstm
    params['use_pos'] = model.use_pos
    params['use_info'] = model.use_info
    params['use_suj'] = model.use_suj
    logger.debug(params)
    return params


def trainer_pairs(train_df, val_df, test_df=None, foldt=''):
    
    if args.double_train:
        _train_df, _val_df, _test_df = train_df.copy(), val_df.copy(), test_df.copy()
        
    multilingual = True if 'multilingual' in args.bert_model else False

    backup_file_path = os.path.join(args.data_folder, \
        f'{foldt}pairs_{args.train_data_name}_{args.strip_title}_{args.repeat}_{args.build_mode}.pickle'.lower())

    if os.path.isfile(backup_file_path) and args.use_backup:
        with open(backup_file_path, 'rb') as handle:
            backup_file = pickle.load(handle)
        train_df = backup_file['train_df']
        train_ref = backup_file['train_ref']
        val_df = backup_file['val_df']
        val_ref = backup_file['val_ref']
        token_to_ix = backup_file['token_to_ix']
        del(backup_file)
        logger.info('-- backup files')
    else:
        train_df, train_ref, token_to_ix = format_into_pairs(train_df, repeat=args.repeat, 
        multilingual=multilingual, random_masking=args.mask, pos_type=args.pos_type, use_suj=args.use_suj)
        val_df, val_ref, token_to_ix = format_into_pairs(val_df, token_to_ix=token_to_ix, 
        multilingual=multilingual, pos_type=args.pos_type, use_suj=args.use_suj)
        with open(backup_file_path, 'wb') as handle:
            pickle.dump({
            'train_df': train_df, 'train_ref': train_ref, 
            'val_df': val_df, 'val_ref': val_ref,
            'token_to_ix': token_to_ix}, 
            handle, protocol=pickle.HIGHEST_PROTOCOL)

    if test_df is not None:
        test_df, test_ref, token_to_ix = format_into_pairs(test_df, token_to_ix=token_to_ix, mode='test', 
        multilingual=multilingual, pos_type=args.pos_type, use_suj=args.use_suj)
    else:
        test_df, test_ref = None, {}
    logger.info(f'max pos ix (+1 for len): {max(token_to_ix.values())}')
    max_len = get_max_sent_len(
        {**train_ref, **val_ref, **test_ref}, max_len=args.max_len, round_up=True)
    logger.debug(f'max_sent_len: {max_len}')
    labels = [i['target'] for i in train_df.values()]+[i['target']for i in val_df.values()]
    class_weights = torch.Tensor(get_class_weight(labels))
    num_classes = class_weights.shape[0]

    tokenizer = get_tokenizer()

    train_dl = create_data_loader(
        data=train_df,
        reference=train_ref,
        tokenizer=tokenizer,
        max_len=max_len,
        batch_size=args.batch_size,
        drop_last=False
    )
    del(train_df)

    if args.val_size>0:
        val_dl = create_data_loader(
            data=val_df,
            reference=val_ref,
            tokenizer=tokenizer,
            max_len=max_len,
            batch_size=args.batch_size,
            drop_last=False
        )
        del(val_df)

    if test_df is not None:
        test_dl = create_data_loader(
            data=test_df,
            reference=test_ref,
            tokenizer=tokenizer,
            max_len=max_len,
            batch_size=args.batch_size,
            drop_last=False
        )
        del(test_df)
    else:
        test_dl = None
    logger.info('-- formatted files')

    model = BiLSTMPairClassifier(
        seq_len=1+(max_len+1)*2,
        n_classes=num_classes,
        lstm_out_dim=args.lstm_out_dim,
        bert_model=args.bert_model,
        dropout_rate=args.dropout_rate,
        use_bilstm=True if 'bilstm' in args.model_name.lower() else False,
        use_pos=args.use_pos,
        use_info=args.use_info,
        use_suj=args.use_suj,
        device=device
    )
    save_json(
        get_model_params(model), 
        f'{args.out_folder}/{args.model_name}/{foldt}{args.model_save_name}.json'
        )
    model = model.to(device)
    logger.info('-- loaded model')

    if args.log_level == 'debug':
        data = next(iter(train_dl))
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        logger.debug(f'input_ids: {input_ids}')
        logger.debug(f'input_ids.shape: {input_ids.shape}')  # batch size x seq length
        logger.debug(f'attention_mask.shape: {attention_mask.shape}')  # batch size x seq length
        # logger.debug(F.softmax(model(input_ids, attention_mask), dim=1))

    optimizer, scheduler = get_zers(model, len(train_dl))
    if args.use_cw:
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)
    results = defaultdict(list)
    best_accuracy, best_epoch = 0,0

    for epoch in range(args.epochs):

        logger.info(f'Epoch {epoch + 1}/{args.epochs}')
        logger.info('-' * 10)

        train_save_path = None if args.train_save_name is None \
            else f'{args.out_folder}/{args.model_name}/{foldt}epoch{epoch}_{args.train_save_name}'

        train_results = train_step(
            model,
            train_dl,
            criterion,
            optimizer,
            device,
            scheduler,
            len(train_dl.dataset),
            train_save_path
        )

        logger.info(
            'Train loss {:.3f}, accuracy {:.3f}, precision {:.3f}, '.format(
                train_results['train_loss'],
                train_results['train_acc'],
                train_results['train_P'])+\
            'recall {:.3f}, f1score {:.3f}'.format(
                train_results['train_R'],
                train_results['train_F1'])
            )

        if args.val_size>0:
            val_results = evaluate_step(
                model,
                val_dl,
                criterion,
                device,
                len(val_dl.dataset),
                f'{args.out_folder}/{args.model_name}/{foldt}epoch{epoch}_{args.val_save_name}'
            )
            logger.info(
                'Val loss {:.3f}, accuracy {:.3f}, precision {:.3f}, '.format(
                    val_results['val_loss'],
                    val_results['val_acc'],
                    val_results['val_P'])+\
                'recall {:.3f}, f1score {:.3f}'.format(
                    val_results['val_R'],
                    val_results['val_F1'])
                )
            val_acc = val_results['val_acc']
        else:
            val_results = {}
            val_acc = -1

        if test_dl is not None:
            _ = evaluate_step(
                model,
                test_dl,
                criterion,
                device,
                len(test_dl.dataset),
                f'{args.out_folder}/{args.model_name}/{foldt}epoch{epoch}_{args.test_save_name}'
            )

        if (args.early_stopping and val_acc>best_accuracy) or \
            ((not args.early_stopping or args.val_size<=0) and epoch+1==args.epochs):
            torch.save(
                model.state_dict(),
                f'{args.out_folder}/{args.model_name}/{foldt}{args.model_save_name}')
            best_accuracy = val_acc
            best_epoch = epoch

    logger.info(f'best epoch: {best_epoch+1} | val_acc: {best_accuracy}')
    logger.info('-- train complete')

    plot({**train_results, **val_results}, \
        f'{args.out_folder}/{args.model_name}/{foldt}{args.plot_save_name}')

    trainer_pairs2doc(best_epoch, folds_list=[foldt])

    if args.double_train:
        # takes last model and continues training it
        # DOES NOT take best model!
        best_epoch = trainer_bydoc(_train_df, _val_df, _test_df, foldt, model)

    if args.classify_by.lower() == 'mix':
        return model, best_epoch
    else:
        return best_epoch


def get_tokenizer():
    if 'albert' in args.bert_model.lower():
        return AlbertTokenizer.from_pretrained(args.bert_model)
    else:
        return BertTokenizer.from_pretrained(args.bert_model)


def get_zers(model, data_size):
    total_steps = data_size * args.epochs
    if args.optimizer.lower() == 'adam':
        optimizer = AdamW(model.parameters(), lr=args.lr_rate, correct_bias=False)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr_rate, momentum=0.9)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    return optimizer, scheduler
    

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
            sysfile=train_save_path, foldt=f'Train_{foldt}'
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


def trainer(train_df, val_df, test_df=None, foldt=''):
    best_epoch = None
    if args.classify_by.lower() == 'pairs':
        best_epoch = trainer_pairs(train_df, val_df, test_df, foldt)
    else:
        logger.warning('No such classification level type is specified yet.')
        raise NotImplementedError
    return best_epoch


def initialise(mode='train'):
    file_mode = args.train_data_name if mode=='train' else args.test_data_name
    _df = open_json(os.path.join(args.data_folder, file_mode), data_format=list)
    logger.info('-- loaded files')
    if args.build_mode:
        _df = _df[0:20]
    _df = clean_data(_df, args.strip_title, args.translate_from)
    logger.info('-- cleaned files')
    return _df


def get_unseen():
    if args.predict or args.run_type == 'predict_pairs':
        test_df = {k:v for k,v in enumerate(initialise(mode='test'))}
    else:
        test_df = None
    return test_df


def run_one():
    _df = initialise()
    test_df = get_unseen()
    train_df, val_df = get_train_test_data(_df, val_size=args.val_size)
    trainer(train_df, val_df, test_df)


def run_kfolds():
    best_epochs = []
    _df = initialise()
    test_df = get_unseen()
    for fold in range(args.folds):
        train_df, val_df = get_k_train_test_data(data=_df, k=args.folds, seed=args.seed, fold=fold)
        best_epoch = trainer(train_df, val_df, test_df, foldt=f'K{fold}_')
        best_epochs.append(best_epoch)
    if args.classify_by.lower() == 'pairs':
        logger.info(f'best epochs per fold: {best_epochs}')
        trainer_pairs2doc(best_epochs)


def main(task):
    task_func = {
        'run_kfolds': run_kfolds,
        'run_one': run_one,
        'predict_pairs': predict_pairs,
        'pairs2doc': trainer_pairs2doc,
        'review': review_validation,
        'study': study_model
    }
    task_func[task]()


if __name__ == "__main__":
    logger.info('-- starting process')
    log_params(args)
    main(args.run_type)
    save_results_to_csv(os.path.join(args.out_folder, args.results_file))
    logger.info('-- complete process')
