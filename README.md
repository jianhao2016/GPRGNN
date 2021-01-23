# GPRGNN

This is the source code for our ICLR2021 paper: [Adaptive Universal Generalized PageRank Graph Neural Network](https://openreview.net/forum?id=n6jl7fLxrP).


<p align="center">
  <img src="https://github.com/jianhao2016/GPRGNN/blob/master/figs/workflow.png" width="600">
</p>

Hidden state feature extraction is performed by a neural networks using individual node features propagated via GPR. Note that both the GPR weights <img src="https://render.githubusercontent.com/render/math?math=\gamma_k"> and parameter set <img src="https://render.githubusercontent.com/render/math?math=\{\theta\}"> of the neural network are learned simultaneously in an end-to-end fashion (as indicated in red).


The learnt GPR weights of the GPR-GNN on real world datasets. Cora is homophilic while Texas is heterophilic (Here, H stands for the level of homophily defined in the main text, Equation (1)). An interesting trend may be observed: For the heterophilic case the weights alternate from positive to negative with dampening amplitudes. The shaded region corresponds to a 95% confidence interval.


<p align="center">
  <img src="https://github.com/jianhao2016/GPRGNN/blob/master/figs/Different_gammas.png" width="600">
</p>

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

To reproduce the results in Table 2 of [our paper](https://openreview.net/forum?id=n6jl7fLxrP) you need to first perform hyperparameter tuning. 
For details of optimization of all models, please refer to section A.9 in Appendix of our paper. Here are the settings for GPRGNN and APPNP:

We choose random walk path lengths with K = 10 and use a 2-layer (MLP) with 64 hidden units for the NN component. For the GPR weights, we use different initializations including PPR with ![equation](http://www.sciweavers.org/upload/Tex2Img_1611352711/render.png), ![equation](http://www.sciweavers.org/upload/Tex2Img_1611352831/render.png) or ![equation](http://www.sciweavers.org/upload/Tex2Img_1611352861/render.png) and the default random initialization in pytorch. Similarly, for APPNP we search the optimal ![equation](http://www.sciweavers.org/upload/Tex2Img_1611352906/render.png). For other hyperparameter tuning, we optimize the learning rate over {0.002, 0.01, 0.05} and weight decay {0.0, 0.0005} for all models. 

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



