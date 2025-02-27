#!/bin/bash
source ~/miniconda3/bin/activate
conda activate mlp

python train_evaluate_image_classification_system.py --batch_size 32   --learning_rate 0.0001   --num_epochs 200   --experiment_name BaseLR0001TH1HU12   --use_gpu True --transformer_heads 1 --hidden_units 12 --dataset_path "/home/s2742733/mlp/MLP_CW/MIT3sec.csv"

python train_evaluate_image_classification_system.py --batch_size 32   --learning_rate 0.0001   --num_epochs 200   --experiment_name BaseLR0001TH1HU24   --use_gpu True --transformer_heads 1 --hidden_units 24 --dataset_path "/home/s2742733/mlp/MLP_CW/MIT3sec.csv"

python train_evaluate_image_classification_system.py --batch_size 32   --learning_rate 0.0001   --num_epochs 200   --experiment_name BaseLR0001TH2HU12   --use_gpu True --transformer_heads 2 --hidden_units 12 --dataset_path "/home/s2742733/mlp/MLP_CW/MIT3sec.csv"

python train_evaluate_image_classification_system.py --batch_size 32   --learning_rate 0.0001   --num_epochs 200   --experiment_name BaseLR0001TH2HU24   --use_gpu True --transformer_heads 2 --hidden_units 24 --dataset_path "/home/s2742733/mlp/MLP_CW/MIT3sec.csv"

python train_evaluate_image_classification_system.py --batch_size 32   --learning_rate 0.0001   --num_epochs 200   --experiment_name BaseLR0001TH3HU12   --use_gpu True --transformer_heads 3 --hidden_units 12 --dataset_path "/home/s2742733/mlp/MLP_CW/MIT3sec.csv"

python train_evaluate_image_classification_system.py --batch_size 32   --learning_rate 0.0001   --num_epochs 200   --experiment_name BaseLR0001TH3HU24   --use_gpu True --transformer_heads 3 --hidden_units 24 --dataset_path "/home/s2742733/mlp/MLP_CW/MIT3sec.csv"

python train_evaluate_image_classification_system.py --batch_size 32   --learning_rate 0.0001   --num_epochs 200   --experiment_name BaseLR0001TH6HU12   --use_gpu True --transformer_heads 6 --hidden_units 12 --dataset_path "/home/s2742733/mlp/MLP_CW/MIT3sec.csv"

python train_evaluate_image_classification_system.py --batch_size 32   --learning_rate 0.0001   --num_epochs 200   --experiment_name BaseLR0001TH6HU24   --use_gpu True --transformer_heads 6 --hidden_units 24 --dataset_path "/home/s2742733/mlp/MLP_CW/MIT3sec.csv"