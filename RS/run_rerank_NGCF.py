import subprocess

# Training args
data_dir = '../data/amz/proc_data'
task_name = 'rerank'
dataset_name = 'amz'
aug_prefix = 'bert_avg'
augment = True

epoch = 50
batch_size = 256
lr = '5e-4'
lr_sched = 'cosine'
weight_decay = 0

# NGCF specific parameters
model = 'NGCF'  # Changed from LightGCN to NGCF
embed_size = 32
final_mlp = '200,80'
convert_arch = '128,32'
dropout = 0.1  # Increased dropout for NGCF
n_layers = 3   # Number of NGCF layers

convert_type = 'HEA'
convert_dropout = 0.0
export_num = 2
specific_export_num = 5
temperature = 0.5

# Run the train process
for batch_size in [256, 512]:
    for lr in ['5e-4', '1e-3']:
        for weight_decay in ['0', '1e-4']:
            for n_layers in [2, 3, 4]:  # Try different numbers of layers
                print(f'Training NGCF with bs={batch_size}, lr={lr}, wd={weight_decay}, layers={n_layers}')
                
                base_args = ['python', '-u', 'main_rerank.py',
                            f'--save_dir=./model/{dataset_name}/{task_name}/{model}/WDA_Emb{embed_size}_epoch{epoch}'
                            f'_bs{batch_size}_lr{lr}_{lr_sched}_cnvt_arch_{convert_arch}_cnvt_type_{convert_type}'
                            f'_eprt_{export_num}_wd{weight_decay}_drop{dropout}_layers{n_layers}' + \
                            f'_hl{final_mlp}_augment_{augment}',
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
                            f'--lr_sched={lr_sched}',
                            f'--weight_decay={weight_decay}',
                            f'--temperature={temperature}',
                            f'--algo={model}',
                            f'--embed_size={embed_size}',
                            f'--export_num={export_num}',
                            f'--specific_export_num={specific_export_num}',
                            f'--final_mlp_arch={final_mlp}',
                            f'--dropout={dropout}',
                            ]
                
                # NGCF specific arguments
                model_specific_args = [
                    f'--n_layers={n_layers}'
                ]
                
                subprocess.run(base_args + model_specific_args)
