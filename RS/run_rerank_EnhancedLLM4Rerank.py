import subprocess

# Training args
data_dir = '../data/amz/proc_data'
task_name = 'rerank'
dataset_name = 'amz'
aug_prefix = 'bert_avg'
augment = True

# Multi-aspect parameters
aspect_weights = ['0.4,0.3,0.3', '0.33,0.33,0.34']  # accuracy, diversity, fairness
diversity_threshold = ['0.7', '0.8']  # similarity threshold for diversity penalty

# Model architecture parameters
model = 'EnhancedLLM4Rerank'
learning_rate = ['0.1', '0.01']
embed_size = 32
final_mlp = '200,80'
n_head = 4
n_layers = 4
ff_dim = 256

# Training parameters
epoch = 50
batch_size = 256
dropout = 0.1  # 증가된 dropout
weight_decay = 0.01  # L2 정규화 추가

# Context parameters
context_dim = 64
max_time_steps = 100

# Run the training process
for lr in learning_rate:
    for weights in aspect_weights:
        for div_thresh in diversity_threshold:
            print(f'Training EnhancedLLM4Rerank with params: '
                  f'lr={lr}, aspect_weights={weights}, diversity_threshold={div_thresh}')
            
            subprocess.run(['python', '-u', 'main_rerank.py',
    f'--save_dir=./model/{dataset_name}/{task_name}/{model}/Enhanced_LLM4Rerank'
    f'_lr{lr}_weights{weights}_divthresh{div_thresh}',
    f'--data_dir={data_dir}',
    f'--augment={augment}',
    f'--aug_prefix={aug_prefix}',
    f'--task={task_name}',
    f'--epoch_num={epoch}',
    f'--batch_size={batch_size}',
    f'--lr={lr}',  # learning_rate를 lr로 변경
    f'--embed_dim={embed_size}',  # embed_size를 embed_dim으로 변경
    f'--llm_heads={n_head}',  # llm_num_heads를 llm_heads로 변경
    f'--llm_layers={n_layers}',  # llm_num_layers를 llm_layers로 변경
    f'--llm_ff_dim={ff_dim}',
    f'--aspect_weights={weights}',
    f'--diversity_threshold={div_thresh}',
    f'--context_dim={context_dim}',
    f'--max_time_steps={max_time_steps}',
    f'--algo={model}'
])