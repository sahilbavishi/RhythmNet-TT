#!/bin/bash

source ~/miniconda3/bin/activate

conda activate mlp

python train_evaluate_image_classification_system.py --batch_size 32 --is_titan True  --is_pretrain True --learning_rate 0.0001   --num_epochs 200   --experiment_name PreTrainTitanModel0001TH6HU24   --use_gpu True --transformer_heads 6 --hidden_units 24 --phi 1 --dataset_path "/home/s2742733/mlp/MLP_CW/PreTrainMIT3sec.csv"

mkdir -p PostTitanModel0001TH6HU24

mkdir -p PostTitanModel0001TH6HU24/saved_models
mkdir -p PostTitanModel0001TH6HU24/result_outputs

cp -v PreTrainTitanModel0001TH6HU24/saved_models/train_model_199.pth PostTitanModel0001TH6HU24/saved_models/train_model_0.pth

cp -v PreTrainTitanModel0001TH6HU24/result_outputs/summary.csv PostTitanModel0001TH6HU24/result_outputs/summary.csv

python train_evaluate_image_classification_system.py --batch_size 32 --is_titan True  --continue_from_epoch 0 --learning_rate 0.0001   --num_epochs 200   --experiment_name cp -v PreTrainTitanModel0001TH6HU24 --use_gpu True --transformer_heads 6 --hidden_units 24 --phi 1 --dataset_path "/home/s2742733/mlp/MLP_CW/MIT3sec.csv"



# python train_evaluate_image_classification_system.py --batch_size 32   --learning_rate 0.0001   --num_epochs 2   --experiment_name PreTrainBaseModel001TH6HU24   --use_gpu True --transformer_heads 6 --hidden_units 24 --phi 0.5 --dataset_path "/home/s2677266/MLP/MLP_CW/PreTrainMIT3sec.csv"

# python train_evaluate_image_classification_system.py --batch_size 32   --learning_rate 0.0001   --num_epochs 300   --experiment_name PreTrainTitanModel  --use_gpu True --transformer_heads 6 --hidden_units 24 --phi 0.5 --dataset_path "/home/s2677266/MLP/MLP_CW/PreTrainMIT3sec.csv"

# python train_evaluate_image_classification_system.py --continue_from_epoch 228 --batch_size 32   --learning_rate 0.0001   --num_epochs 300   --experiment_name UsingPretrainedBaselinePlease  --use_gpu True --transformer_heads 6 --hidden_units 24 --phi 0.5 --dataset_path "/home/s2677266/MLP/MLP_CW/MIT3sec.csv"


# python train_evaluate_image_classification_system.py --continue_from_epoch 2 --batch_size 32   --learning_rate 0.0001   --num_epochs 4   --experiment_name PreTrainBaseModel001TH6HU24   --use_gpu True --transformer_heads 6 --hidden_units 24 --phi 0.5 --dataset_path "/home/s2677266/MLP/MLP_CW/MIT3sec.csv"sbat