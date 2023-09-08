# virt
Visual RetNet for Cifar-10
# 1. Setup Development Environment
This project requires Pytorch 2.x version. Install dependencies:
```bash
pyyam
tensorboard
transformers
einops
```
## 1.2. Prepare datasets
We use cifar-10 dataset. Let's download and extract it:
```bash
wget cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
```

## 1.3. Initial Run
```bash
ssh ps@192.168.2.31 # pwd: chen652
cd /mnt/10t/yantao/adev/cv/virt
python app_main.py --run_mode 1 # train
python app_main.py --run_mode 2 # eval
```


# Reference
| seq | url | title |
| --- | --- | --- |
| 1 | Retentive Network: A Successor to Transformer for Large Language Models | https://arxiv.org/pdf/2307.08621.pdf |
| 2 | RetNet implementation | https://github.com/syncdoth/RetNet |


