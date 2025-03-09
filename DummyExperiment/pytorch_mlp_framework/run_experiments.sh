#!/bin/bash
source ~/miniconda3/bin/activate
conda activate mlp

python train_evaluate_image_classification_system.py --batch_size 32   --learning_rate 0.0001   --num_epochs 400   --experiment_name SmallTitanLR0001TH6HU24   --use_gpu True --transformer_heads 6 --hidden_units 24 --phi 0.5 --dataset_path "/home/s2742733/mlp/MLP_CW/MIT3sec.csv"

python train_evaluate_image_classification_system.py --batch_size 32   --learning_rate 0.0001   --num_epochs 400   --experiment_name SmallBaseLR0001TH6HU24   --use_gpu True --transformer_heads 6 --hidden_units 24 --phi 0.5 --dataset_path "/home/s2742733/mlp/MLP_CW/MIT3sec.csv"