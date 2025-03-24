#!/bin/bash

source ~/miniconda3/bin/activate

conda activate mlp

#python physionet_tester.py --experiment_name PostBaseModel0001TH6HU24 --epoch -1 --MLP_path "/home/s2742733/mlp/" --is_titan False --timespan 3

python physionet_tester.py --experiment_name PostTitanModel0001TH6HU24 --epoch -1 --MLP_path "/home/s2742733/mlp/" --is_titan True --timespan 3