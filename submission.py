"""
format doc_clus in csv into CASE2021 submission format in json
"""
import pandas as pd
import ast
import json
import os
import re
import shutil


# open pred
data = {}
en_output, es_output, pt_output = [], [], []
sub_folder_name = 'bilstm_pairs_case_wpos_winfo_wsuj'
folder_name = f'case2021/{sub_folder_name}'
dir_path = os.path.dirname(os.path.realpath(__file__))
pred_df = pd.read_csv(f'outs/{folder_name}/_doc_ors_test_res.csv')

# format pred acc to original file
for k, line in enumerate(open('data/all_en_test.json', encoding='utf-8')):
    _v = json.loads(line)
        
    v = {'id': _v['id']}
    
    if k not in list(pred_df['id']):
        continue
    
    clusters = []
    predictions = ast.literal_eval(pred_df[pred_df['id']==k]['pred'].item())

    for clus in set(predictions):
        clusters.append([
            s for s, p in zip(_v['sentence_no'], predictions) if p==clus
        ])
    
    v['pred_clusters'] = clusters

    if _v['tag']=='en':
        en_output.append(v)
    elif _v['tag']=='es':
        es_output.append(v)
    else:
        pt_output.append(v)
    
# write to file
def save_and_zip(output, ext):
    new_folder_name = os.path.join(dir_path, f'outs/{folder_name}/{ext}')
    new_folder_name = re.sub(r'/|\\', re.escape(os.sep), new_folder_name)
    print(new_folder_name)
    os.mkdir(new_folder_name)
    with open(os.path.join(new_folder_name, 'submission.json'), "w") as outfile:
        for obj in output:
            outfile.write(json.dumps(obj) + '\n')
    shutil.make_archive(f'outs/{folder_name}/{sub_folder_name}__{ext}', 'zip', new_folder_name)

save_and_zip(en_output, 'en')
save_and_zip(es_output, 'es')
save_and_zip(pt_output, 'pt')




