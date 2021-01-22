# GPRGNN

This is the source code for our ICLR2021 paper: [Adaptive Universal Generalized PageRank Graph Neural Network](https://openreview.net/forum?id=n6jl7fLxrP).

# Requirement:
```
pytorch
pytorch-geometric
numpy
```

# Run experiment with Cora:

go to folder `src`
```
python train_model.py --RPMAX 2 \
        --net GPRGNN \
        --train_rate 0.025 \
        --val_rate 0.025 \
        --dataset cora 
```

# Create cSBM dataset:
go to folder `src`
```
source create_cSBM_dataset.sh
```
 
# Repreduce results in Table 2:

To reproduce the results in Table 2 of [our paper](https://openreview.net/forum?id=n6jl7fLxrP) you need to first perform hyperparameter tuning. We choose random walk path lengths with K = 10 and use a 2-layer (MLP) with 64 hidden units for the NN component. For the GPR weights, we use different initializations including PPR with ![equation](http://www.sciweavers.org/upload/Tex2Img_1611352711/render.png), ![equation](http://www.sciweavers.org/upload/Tex2Img_1611352831/render.png) or ![equation](http://www.sciweavers.org/upload/Tex2Img_1611352861/render.png) and the default random initialization in pytorch. Similarly, for APPNP we search the optimal ![equation](http://www.sciweavers.org/upload/Tex2Img_1611352906/render.png). For other hyperparameter tuning, we optimize the learning rate over {0.002, 0.01, 0.05} and weight decay {0.0, 0.0005} for all models. 

# Citation
Please cite our paper if you use this code in your own work:
```latex
@inproceedings{
chien2021adaptive,
title={Adaptive Universal Generalized PageRank Graph Neural Network},
author={Eli Chien and Jianhao Peng and Pan Li and Olgica Milenkovic},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=n6jl7fLxrP}
}
```

Feel free to email us(jianhao2@illinois.edu, ichien3@illinois.edu) if you have any further questions. 



