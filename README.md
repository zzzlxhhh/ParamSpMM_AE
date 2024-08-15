# ParamSpMM_artifacts

The artifact for ParamSpMM: Empowering Graph Neural Networks with High-Performance Parameterized Sparse Matrix-Matrix Multiplication on GPUs.

## environment requirements

```
OS: Ubuntu 20.04.5 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
CUDA runtime version: 11.6.124
```
environment setup
```
# install libraries for compiling rabbit reordering
apt-get update 
apt-get install -y libboost-all-dev libgoogle-perftools-dev
# create ParamSpMM conda enviroment
conda create -n param python=3.9
# install pytorch 1.13
conda activate param
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
# install DGL
conda install -c dglteam dgl-cuda11.6==0.9.1post1
# install neccessary libraries
pip install ssgetpy
pip install pandas
pip install ogb
pip install scikit-learn==1.1.3
pip install ipykernel
pip install seaborn
```

All the following scripts are executed under Anaconda environment `param`

## Use ParamSpMM as a tool
  
- In directory `/script`: run `bash build_param.sh` to build and bind `ParamSpMM` and `ParamSDDMM` into pytorch. The ML models for SpMM-decider are already available in directory `/model`. 

- To run `ParamSpMM` with different column dimensions:

  ```python
     # path: the path of *.mtx file
     # mtx_name: the name of the sparse matrix
     # dim_list: the column dimensions of dense input matrices 
     python decider.py --path ./toy_dataset/ --mtx_name pubmed --dim_list 16
  ```
- To run GCN:
  ```python
     # num_classes: dimension of output classes
     # num_features: dimension of features
     # num_epoches: training epoches
     # hidden_list: hidden layer size 
     python GNN.py --path ./toy_dataset/ --mtx_name pubmed --num_classes 16 \
        --num_features 16 --num_epoches 200 --hidden_list 16 --num_layers 5
  ```
- To run GIN:
   ```python
      python GNN.py --net GIN --path ./toy_dataset/ --mtx_name pubmed --num_classes 16 \
        --num_features 16 --hidden_list 32 --num_layers 5 
   ```
- To run AGNN embedded with `ParamSpMM` and `ParamSDDMM`:
   ```python
      python GNN.py --net AGNN --path ./toy_dataset/ --mtx_name pubmed --num_classes 16 \
        --num_features 16 --hidden_list 32 --num_layers 5 
   ```

## Rebuild SpMM-decider from scratch
Despite we provide the trained ML models of SpMM-decider in directory `/models`, you can still train your own SpMM-decider based on the following steps:
### step 0: prepare graph datasets

For easy reproduction, we provide the processed matrix datasets [Here](https://drive.google.com/file/d/1kZe-yFP0sIlzVc7aXXEt2hGuBmrjFdiU/view?usp=drive_link). The matrices used for ParamSpMM evaluation and SpMM-decieder training are in `./all-mtx`. The rabbit-reordered matrices are in `./all-mtx/rabbit`. The names of all the 202 matrices and their features are in `mtx_info.csv`. The 6 large graphs for GNN training are in `./OGB`. The rabbit-reordered matrices are in `./OGB/rabbit`. After finishing downloading, try to unzip the file with:

```bash
bash 0_unzip.sh
```

### step 1: construct the training set for SpMM-decider

In this step, we construct the training set for SpMM-decider by collecting the optimal configurations of `ParamSpMM` across all the matrices.  Baseline SpMM libraries including `cuSPARSE` and `GE-SpMM` are also evaluated in this part.

In directory `/script`:  

- run `bash 2_construct_training_set.sh`
  As a result, the performances of `ParamSpMM`, `cuSPARSE` and `GE-SpMM` are written to `ParamSpMM-log`. For example,  
  - `dim16.csv` stores the throughputs of `ParamSpMM` under different configurations from reduced selection space. 
  - `dim16_OP_SpMM.csv` contains the matrices information and their optimal `ParamSpMM` configurations, which are used as the training set for SpMM-decider when `dim=16`.
  - `dim16_basline.csv` contains the throughput of `cuSPARSE` and `GE-SpMM` when `dim=16`.

The throughputs when $dim \in \{16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256\}$ are evaluated.


### step 2: Model training, the performance of SpMM-decider
We choose random forest models for predicting the optimal configuration of `ParamSpMM` in SpMM-decider. 
The training process in `rnd_tree.ipynb`. The trained random forests models are then serialized pickle files and stored in `./models`, which can be used repeatedly. 

