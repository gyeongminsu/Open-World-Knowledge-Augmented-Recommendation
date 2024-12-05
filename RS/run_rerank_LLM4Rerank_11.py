import subprocess

# Training args
data_dir = '../data/amz/proc_data'
task_name = 'rerank'
dataset_name = 'amz'
aug_prefix = 'bert_avg'
augment = True

# Base hyperparameters
epoch = 50
embed_size = 32
final_mlp = '200,80'
dropout = 0.0
convert_type = 'HEA'
convert_arch = '128,32'

# LLM4Rerank specific parameters
llm_heads = 8
llm_layers = 6
llm_ff_dim = 512

model = 'LLM4Rerank_1'

# Hyperparameter search grid
batch_sizes = [256, 512, 128, 64]
learning_rates = ['5e-4', '1e-3']
weight_decays = ['0', '1e-4', '1e-3']
llm_heads_options = [8]

for batch_size in batch_sizes:
    for lr in learning_rates:
        for weight_decay in weight_decays:
            for num_heads in llm_heads_options:
                print(f'--------------- Testing with bs={batch_size}, lr={lr}, wd={weight_decay}, heads={num_heads} -----------')
                
                base_args = [
                    'python', '-u', 'main_rerank.py',
                    f'--save_dir=./model/{dataset_name}/{task_name}/{model}/LLM_Emb{embed_size}_epoch{epoch}'
                    f'_bs{batch_size}_lr{lr}_heads{num_heads}_wd{weight_decay}',
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
                
                # LLM specific arguments
                llm_args = [
                    f'--llm_heads={num_heads}',
                    f'--llm_layers={llm_layers}',
                    f'--llm_ff_dim={llm_ff_dim}',
                ]

                subprocess.run(base_args + llm_args)