# Precipitation-Nowcasting

This is an easy-to-understand implementation of ConvLSTM model(fisrt proposed by [Xinjian Shi et al.])(https://arxiv.org/abs/1506.04214https://arxiv.org/abs/1506.04214) in a real-world precipitation nowcasting problem with Pytorch. Here presents the guidance on how to run this project by yourself. Have fun!

## DATA
#### Two open-sourced datasets are available for training and testing in this project.

1. A pre-masked radar datasets.(Included in the package)     
2. Tianchi CNKI 2017 dataset（Provided by Shenzhen Meteorological Bureau）.This dataset is not included yet. However, You can download the datasets [here](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.6d453864enogCW&raceId=231596)

## Getting Started
### Prerequisites  
Environment:   
* Win10 or Win7  
* Anaconda 3-5.1  
* Python 3.6  
* CUDA 8(BOTH 0.3.1 & 1.4.0) or CUDA10.1(ONLY FOR 1.4.0)

### Installing
1. Install CUDA8(BOTH 0.3.1 & 1.4.0) or CUDA10.1(ONLY FOR 1.4.0)

2. Download and install Anaconda environment  

3. Install an environment
```
  \# conda create -n project python=3.6 
  \# conda create -n old python=3.6 
  
```
4. Activate your new-built environemt and install Pytorch and torchvision (For Nowcasting)
```
  \# activate project 
  \# conda install -c peterjc123 pytorch (WIN10)
  \# conda install -c peterjc123 pytorch cuda80 (WIN7)
  \# pip install torchvision===0.2.1 -f https://download.pytorch.org/whl/torch_stable.html
  \# pip install arrow
  \# pip install pillow===6.0.0
  \# pip install tqdm
  \# pip install colorama
  \# conda deactivate
```
5. Activate your new-built environemt and install Pytorch and torchvision (For Training)
```
  \# activate old 
  \# pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
  \# pip install arrow
  \# pip install pillow
  \# pip install tqdm
  \# pip install colorama
  \# conda deactivate
```

### Train the model 

1. Download the all package and unpack it   
Note: you also need to unpack the files in the original `data` directory before training  

2. Train the model 
```
  Python training.py
```

### Running the test 

Run the test.py with the command. 
```
  python test.py  
```
Evaluate your model's performance by running 
```
  python evaluate.py
```

### Authors  
     cxxixi
     pqx
     Amy Hsiao
     Thomas Liu 
     Garry Lai

## Notes
1. [`Notes on ConvLSTM`](https://github.com/cxxixi/Precipitation-Nowcasting/issues/1)
