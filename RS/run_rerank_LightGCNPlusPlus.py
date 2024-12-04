import subprocess

# Training args
data_dir = '../data/amz/proc_data'
task_name = 'rerank'
dataset_name = 'amz'
aug_prefix = 'bert_avg'
augment = True

epoch = 50
embed_size = 32
final_mlp = '200,80'
convert_arch = '128,32'
num_cross_layers = 3
convert_type = 'HEA'
convert_dropout = 0.0
temperature = 0.5
n_layers = 4

model = 'LightGCNPlusPlus'  # 모델명 변경

# Hyperparameter grid search
configurations = [
    {
        'batch_size': bs,
        'lr': lr,
        'weight_decay': wd,
        'export_config': (exp_num, spec_num)
    }
    for bs in [256, 512, 128, 64]
    for lr in ['5e-4', '1e-3']
    for wd in ['0', '1e-4', '1e-3']
    for exp_num, spec_num in [(2, 5), (3, 4), (2, 6)]
]

for config in configurations:
    batch_size = config['batch_size']
    lr = config['lr']
    weight_decay = config['weight_decay']
    export_num, specific_export_num = config['export_config']

    print('---------------bs,lr,wd,epoch, export share/spcf, convert arch ----------', 
          batch_size, lr, epoch, weight_decay, export_num, specific_export_num, 
          convert_arch, model)
    
    # LightGCN++ specific parameters
    model_name = f"{model}_nl{n_layers}"
    
    save_dir = (f'./model/{dataset_name}/{task_name}/{model_name}/'
                f'WDA_Emb{embed_size}_epoch{epoch}_bs{batch_size}_lr{lr}_'
                f'cnvt_arch_{convert_arch}_cnvt_type_{convert_type}_'
                f'eprt_{export_num}_wd{weight_decay}_drop{0.0}'
                f'_hl{final_mlp}_cl{num_cross_layers}_augment_{augment}')

    base_args = [
        'python', '-u', 'main_rerank.py',
        f'--save_dir={save_dir}',
        f'--data_dir={data_dir}',
        f'--augment={augment}',
        f'--aug_prefix={aug_prefix}',
        f'--task={task_name}',
        f'--convert_arch={convert_arch}',
        f'--convert_type={convert_type}',
        f'--convert_dropout={convert_dropout}',
        f'--epoch_num={epoch}',
        f'--batch_size={batch_size}',
        f'--lr={lr}',
        f'--lr_sched=cosine',
        f'--weight_decay={weight_decay}',
        f'--temperature={temperature}',
        f'--algo={model}',
        f'--embed_dim={embed_size}',
        f'--export_num={export_num}',
        f'--specific_export_num={specific_export_num}',
        f'--final_mlp_arch={final_mlp}',
        f'--dropout=0.0',
    ]
    
    # LightGCN++ specific arguments
    model_specific_args = [
        f'--n_layers={n_layers}',
        '--norm_scale=1.0',
        '--layer_weight_init=1.0',
        '--neighbor_weight_init=1.0'
    ]

    subprocess.run(base_args + model_specific_args)