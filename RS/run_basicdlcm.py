
import subprocess

# Training args
data_dir = '../data/amz/proc_data'
task_name = 'rerank'
dataset_name = 'amz'

# No augmentation for basic models
augment = False
aug_prefix = ''

# Base hyperparameters
epoch = 50
embed_size = 32
final_mlp = '200,80'
dropout = 0.0
convert_type = 'HEA'
convert_arch = '128,32'

# Model specific parameters
llm_heads = 8
llm_layers = 6
llm_ff_dim = 512
n_layers = 3  # For GCN based models

# Models to test
models = ['BasicDLCM']

# Hyperparameter search grid
batch_sizes = [256, 512, 128]
learning_rates = ['5e-4', '1e-3']
weight_decays = ['0', '1e-4']

for model in models:
    print(f'=== Running experiments for {model} ===')
    
    for batch_size in batch_sizes:
        for lr in learning_rates:
            for weight_decay in weight_decays:
                print(f'Testing with bs={batch_size}, lr={lr}, wd={weight_decay}')
                
                base_args = [
                    'python', '-u', 'main_rerank.py',
                    f'--save_dir=./model/{dataset_name}/{task_name}/{model}/Basic_Emb{embed_size}_epoch{epoch}'
                    f'_bs{batch_size}_lr{lr}_wd{weight_decay}',
                    f'--data_dir={data_dir}',
                    f'--augment={augment}',
                    f'--aug_prefix={aug_prefix}',
                    f'--task={task_name}',
                    f'--epoch_num={epoch}',
                    f'--batch_size={batch_size}',
                    f'--lr={lr}',
                    f'--weight_decay={weight_decay}',
                    f'--algo={model}',
                    f'--embed_dim={embed_size}',
                    f'--final_mlp_arch={final_mlp}',
                    f'--dropout={dropout}',
                ]

                # Add model specific arguments
                if model in ['BasicLLM4Rerank']:
                    model_args = [
                        f'--llm_heads={llm_heads}',
                        f'--llm_layers={llm_layers}',
                        f'--llm_ff_dim={llm_ff_dim}',
                    ]
                elif model in ['BasicLightGCN', 'BasicNGCF']:
                    model_args = [
                        f'--n_layers={n_layers}',
                    ]
                else:  # BasicDLCM
                    model_args = []

                subprocess.run(base_args + model_args)

                print(f'Completed run for {model} with bs={batch_size}, lr={lr}, wd={weight_decay}')
    
    print(f'=== Completed all experiments for {model} ===')