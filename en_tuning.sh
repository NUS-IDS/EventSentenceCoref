# CASE 2021
##### on combined train data that is pretranslated into english #####
# kfold for tuning (to find ideal #epochs)

### base
python3 -W ignore main.py\
    --classify_by pairs --model_name bilstm_pairs_case_en\
    --run_type run_kfolds --strip_title false --use_info false --use_pos false\
    --train_data_name all_en_train.json --test_data_name all_en_test.json --folds 5\
    --repeat false --cuda_device 1 --predict true --use_backup false

### wpos
python3 -W ignore main.py\
    --classify_by pairs --model_name bilstm_pairs_case_en_wpos\
    --run_type run_kfolds --strip_title false --use_info false --use_pos true\
    --train_data_name all_en_train.json --test_data_name all_en_test.json --folds 5\
    --repeat false --cuda_device 1 --predict true --use_backup false

### wpos + winfo
python3 -W ignore main.py\
    --classify_by pairs --model_name bilstm_pairs_case_en_wpos_winfo\
    --run_type run_kfolds --strip_title false --use_info true --use_pos true\
    --train_data_name all_en_train.json --test_data_name all_en_test.json --folds 5\
    --repeat false --cuda_device 1 --predict true --use_backup false

### wpos + winfo + wsuj
python3 -W ignore main.py\
    --classify_by pairs --model_name bilstm_pairs_case_en_wpos_winfo_wsuj\
    --run_type run_kfolds --strip_title false --use_info true --use_pos true --use_suj true\
    --train_data_name all_en_train.json --test_data_name all_en_test.json --folds 5\
    --repeat false --cuda_device 1 --predict true --use_backup false