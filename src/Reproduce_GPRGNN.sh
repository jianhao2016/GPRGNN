#! /bin/sh
#
# This script is to reproduce our results in Table 2.
RPMAX=100

# Below is for homophily datasets, sparse split

python train_model.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.025 \
        --val_rate 0.025 \
        --dataset cora \
        --lr 0.01 \
        --alpha 0.1 

python train_model.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.025 \
        --val_rate 0.025 \
        --dataset citeseer \
        --lr 0.01 \
        --alpha 0.1

python train_model.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.025 \
        --val_rate 0.025 \
        --dataset pubmed \
        --lr 0.05 \
        --alpha 0.2 
        
python train_model.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.025 \
        --val_rate 0.025 \
        --dataset computers \
        --lr 0.05 \
        --alpha 0.5 \
        --weight_decay 0.0
        
python train_model.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.025 \
        --val_rate 0.025 \
        --dataset photo \
        --lr 0.01 \
        --alpha 0.5 \
        --weight_decay 0.0

# Below is for heterophily datasets, dense split

python train_model.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.6 \
        --val_rate 0.2 \
        --dataset chameleon \
        --lr 0.05 \
        --alpha 1.0 \
        --weight_decay 0.0 \
        --dprate 0.7 

python train_model.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.6 \
        --val_rate 0.2 \
        --dataset film \
        --lr 0.01 \
        --alpha 0.9 \
        --weight_decay 0.0 

python train_model.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.6 \
        --val_rate 0.2 \
        --dataset squirrel \
        --lr 0.05 \
        --alpha 0.0 \
        --weight_decay 0.0 \
        --dprate 0.7 
        
python train_model.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.6 \
        --val_rate 0.2 \
        --dataset texas \
        --lr 0.05 \
        --alpha 1.0 \
        
python train_model.py --RPMAX $RPMAX \
        --net GPRGNN \
        --train_rate 0.6 \
        --val_rate 0.2 \
        --dataset cornell \
        --lr 0.05 \
        --alpha 0.9 

