# Collabrative Filter

## Depencence
- Python 2.7
    - Tensorflow 1.1 
    - Keras 2

## Run
- Put `train_sub_txt.txt` in `data\`
```
$ROOT
├── clean.sh
├── data
│   └── train_sub_txt.txt
├── output
├── README.md
└── src
    ├── config.py
    ├── data.py
    ├── ensemble.py
    ├── grid_search.py
    ├── logger.py
    ├── model.py
    └── utils.py
```
- Run `ensemble.py`, generate `res.txt` in `output\`

## TODO list
- [x] Ensemble SVD and DaulNet 
- [ ] Whether to Earlystop 
- [ ] Whether to balance training dataset (Aug+Shuffle)
- [x] SVD 
    - Grid search 
        - latent dim 
        - regularize hyperparameter 
        - Dropout
        - Data Augmentation 
    - accuracy on the fly
- [x] DaulNet
    - Grid search
- TFMF tell me cold start 
- Surprise tell me matplotlib new backend 
- last result is [('deep_cf_bfdzizbbvz', 0.44244604316546765), ('deep_cf_oazxdggvis', 0.44308082945408378), ('deep_cf_fgqeyhawyb', 0.44436118396403146), ('deep_cf_xzylamykpm', 0.43218433870363432), ('deep_cf_mknxdfytei', 0.43368302735106784), ('deep_cf_nnixrbgavx', 0.43611839640314726)]
- combine final result
    - are the perfomance on test set improve? 
    - are the results consitence?
- gridsearch? checkpoint for svd?

# Reference
- `SVD` : https://github.com/mesuvash/TFMF/blob/master/TFMF.ipynb 
- `DaulNet`/`DeepCF` re-implement from Xiong Y, Lin D, Niu H, et al. Collaborative Deep Embedding via Dual Networks[J]. 2016. (With Some Modifications)
