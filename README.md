# EGAN : Generative Adversarial Network for Skin Lesion Segmentation

Automatic lesion segmentation is a critical computer aided diagnosis (CAD) tool vital in ensuring effective treatment. Computer-aided diagnosis of such skin cancer on dermoscopic images can significantly reduce the clinicians’ workload and improve diagnostic accuracy. This paper proposes an adversarial learning-based segmentation framework that leverages the adversarial learning-based framework (EGAN) for skin lesion segmentation. Specifically, this framework integrates two modules: The segmentation module and the discriminator module. 

![gan_architecture](https://user-images.githubusercontent.com/50418503/156883185-1da2dd2c-4635-479d-8e97-8f328a1fa3b3.png)


## Getting Started

### Install Requirements
tensorflow 2.x
keras=2.2.4
opencv
tqdm
scikit-image

### Prerequisites
GPU
CUDA

### Running Evaluation
- Clone this repo:
```bash
git clone https://github.com/shubhaminnani/EGAN.git
cd EGAN
```
### To reproduce the results for the rank in SKIN challenge in ISIC 2018, please do
``` bash
python predict.py 0 # 0 is the avaliable GPU id, change is neccesary
```

### Running Training for SKIN ISIC 2018 dataset

Remember to check/change the data path and weight path

```bash
python train_DGS.py 0
python test_DGS.py 0
```

### Citation
```
@article{,
  journal={},
  title={EGAN},
  author={},
  year={},
  volume={},
  number={},
  pages={},
  publisher={},
  doi={},
  }
```  
