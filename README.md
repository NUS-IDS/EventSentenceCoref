# EventSentenceCoref
We worked on the Event sentence coreference identification (ESCI) Shared Task from Challenges and Applications of Automated Extraction of Socio-political Events from Text (CASE) @ ACL-IJCNLP 2021.
Our paper is titled "NUS-IDS at CASE 2021 Task 1: Improving Multilingual Event Sentence Coreference Identification With Linguistic Information".

### Abstract
Event Sentence Coreference Identification (ESCI) aims to cluster event sentences that refer to the same event together for information extraction. We describe our ESCI solution developed for the ACL-CASE 2021 shared tasks on the detection and classification of socio-political and crisis event information in a multilingual setting. For a given article, our proposed pipeline comprises of an accurate sentence pair classifier that identifies coreferent sentence pairs and subsequently uses these predicted probabilities to cluster sentences into groups. Sentence pair representations are constructed from fine-tuned BERT embeddings plus POS embeddings fed through a BiLSTM model, and combined with linguistic-based lexical and semantic similarities between sentences. Our best models ranked 2nd, 1st and 2nd and obtained CoNLL F1 scores of 81.20%, 93.03%, 83.15% for the English, Portuguese and Spanish test sets respectively in the ACL-CASE 2021 competition.

# Prerequisites

### Dependencies
Please refer to the file under `conda_linux_setup.txt` to install dependencies. We worked in a Conda environment on Linux OS. You might need to adjust the Pytorch dependencies according to your system too. You may find compatible Pytorch version on [Pytorch's website](https://pytorch.org/get-started/locally/).

### Dataset
Our dataset is from [CASE 2021](https://github.com/emerging-welfare/case-2021-shared-task) for Shared Task 1 Subtask 3. Please contact the organisers for access. Save the data files under a "data/" folder in this directory. There should be 3 files for train and test each. For example, for train, we have English (`en-train.json`), Portuguese (`pr-train.json`) and Spanish (`es-train.json`) data files.

In our implementation, we merged the three train sets together and three test sets together before running our code. We created two types using Jupyter Notebooks, available under the folder `notebook/`: <br>
1. `CombineAll.ipynb`: All articles are merged into one file forming our "Multilingual train set" (`all_train.json`). We will use language-agnostic models on this corpus.<br>
2. `CombineAll&EnglishAll.ipynb`: All non-English sentences are all translated into English using Google Translate API and are merged into one file forming our "English train set" (`all_en_train.json`). We will use English language models on this corpus.<br>

Similar steps are used to create test sets `all_test.json` and `all_en_test.json`.

### Extended features
We created the "Extended similarities" features, which for now, are created as a standalone module separate from our pipeline. The code is provided under `src/extsimfeats`. We also upload the feature vector per article in pickle files under the folder `data/` for use.

# Training & Predicting
After obtaining the datasets, you can run in command line `python main.py` directly if your defaults are set properly. If not, to run KFolds on our best English BERT model that includes all created features introduced in our paper, run the following command:
``` 
    python3 -W ignore main.py\
    --classify_by pairs --model_name bilstm_pairs_case_en_wpos_winfo_wsuj\
    --run_type run_kfolds --strip_title false --use_info true --use_pos true --use_suj true\
    --train_data_name all_en_train.json --test_data_name all_en_test.json --folds 5\
    --repeat false --cuda_device 1 --predict true --use_backup false
```
We also provided some example commands under `en_tuning.sh` to create validation results found under Table 2 in our paper.

