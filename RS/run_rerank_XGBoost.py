import subprocess

# Training args
data_dir = '../data/amz/proc_data'
task_name = 'rerank'
dataset_name = 'amz'
aug_prefix = 'bert_avg'
augment = True

# XGBoost specific parameters
model = 'XGBoostReranker'
learning_rate = ['0.1', '0.01']
max_depth = ['6', '8']
n_estimators = ['100', '200']
subsample = ['0.8']
colsample_bytree = ['0.8']

# Common parameters
embed_size = 32
final_mlp = '200,80'
convert_arch = '128,32'
convert_type = 'HEA'
convert_dropout = 0.0
export_num = 2
specific_export_num = 5
temperature = 0.5
epoch = 50
batch_size = 256
weight_decay = 0
dropout = 0.0

# Run the train process with XGBoost hyperparameter tuning
for lr in learning_rate:
    for depth in max_depth:
        for n_est in n_estimators:
            for sub in subsample:
                for col_sample in colsample_bytree:
                    print(f'Training XGBoost with params: lr={lr}, depth={depth}, n_est={n_est}, '
                          f'subsample={sub}, colsample={col_sample}')
                    
                    subprocess.run(['python', '-u', 'main_rerank.py',
                        f'--save_dir=./model/{dataset_name}/{task_name}/{model}/XGB_Emb{embed_size}_epoch{epoch}'
                        f'_lr{lr}_depth{depth}_nest{n_est}_sub{sub}_col{col_sample}',
                        f'--data_dir={data_dir}',
                        f'--augment={augment}',
                        f'--aug_prefix={aug_prefix}',
                        f'--task={task_name}',
                        f'--convert_arch={convert_arch}',
                        f'--convert_type={convert_type}',
                        f'--convert_dropout={convert_dropout}',
                        f'--epoch_num={epoch}',
                        f'--batch_size={batch_size}',
                        f'--learning_rate={lr}',
                        f'--max_depth={depth}',
                        f'--n_estimators={n_est}',
                        f'--subsample={sub}',
                        f'--colsample_bytree={col_sample}',
                        f'--temperature={temperature}',
                        f'--algo={model}',
                        f'--embed_size={embed_size}',
                        f'--export_num={export_num}',
                        f'--specific_export_num={specific_export_num}',
                        f'--final_mlp_arch={final_mlp}',
                        f'--dropout={dropout}',
                    ])