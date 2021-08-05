#! /bin/sh
#
# create_cSBM_dataset.sh

python cSBM_dataset.py --phi 0.5 \
    --name cSBM_phi_0.5 \
    --root ../data/ \
    --num_nodes 5000 \
    --num_features 2000 \
    --avg_degree 5 \
    --epsilon 3.25 
