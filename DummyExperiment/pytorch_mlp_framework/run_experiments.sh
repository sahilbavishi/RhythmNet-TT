#!/bin/bash

# conda activate mlp

# python train_evaluate_image_classification_system.py --batch_size 32   --learning_rate 0.0001   --num_epochs 2   --experiment_name PreTrainBaseModel001TH6HU24   --use_gpu True --transformer_heads 6 --hidden_units 24 --phi 0.5 --dataset_path "/home/s2677266/MLP/MLP_CW/PreTrainMIT3sec.csv"

python train_evaluate_image_classification_system.py --continue_from_epoch 1 --batch_size 32   --learning_rate 0.0001   --num_epochs 300   --experiment_name UsingPretrained   --use_gpu True --transformer_heads 6 --hidden_units 24 --phi 0.5 --dataset_path "/home/s2677266/MLP/MLP_CW/MIT3sec.csv"


# python train_evaluate_image_classification_system.py --continue_from_epoch 2 --batch_size 32   --learning_rate 0.0001   --num_epochs 4   --experiment_name PreTrainBaseModel001TH6HU24   --use_gpu True --transformer_heads 6 --hidden_units 24 --phi 0.5 --dataset_path "/home/s2677266/MLP/MLP_CW/MIT3sec.csv"sbat