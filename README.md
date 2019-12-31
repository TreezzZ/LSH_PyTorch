# Similarity search in high dimensions via hashing

## REQUIREMENTS
`pip install -r requirements.txt`

1. pytorch >= 1.0
2. loguru

## DATASETS
[cifar10-gist.mat](https://pan.baidu.com/s/1qE9KiAOTNs5ORn_WoDDwUg) password: umb6

[cifar-10_alexnet.t](https://pan.baidu.com/s/1ciJIYGCfS3m0marQvatNjQ) password: f1b7

[nus-wide-tc21_alexnet.t](https://pan.baidu.com/s/1YglFwoxB-3j7xTEyAc8ykw) password: vfeu

[imagenet-tc100_alexnet.t](https://pan.baidu.com/s/1ayv4wdtCOzEDsJy01SjRew) password: 6w5i

## USAGE
```
usage: run.py [-h] [--dataset DATASET] [--root ROOT]
              [--code-length CODE_LENGTH] [--topk TOPK] [--gpu GPU]

LSH_PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset name.
  --root ROOT           Path of dataset
  --code-length CODE_LENGTH
                        Binary hash code length.(default:
                        8,16,24,32,48,64,96,128)
  --topk TOPK           Calculate top k data map.(default: all)
  --gpu GPU             Using gpu.(default: False)
```

## EXPERIMENTS
cifar10-gist dataset. 1000 query images, 59000 retrieval images, MAP@ALL.

cifar-10-alexnet dataset. Alexnet features, 1000 query images, 59000 retrieval images, MAP@ALL.

nus-wide-tc21-alexnet dataset. Alexnet features, top 21 classes, 2100 query images, 193734 retrieval images, MAP@5000.

imagenet-tc100-alexnet dataset. Alexnet features, top 100 classes, 5000 query images, 130000 retrieval images, MAP@1000.


   Bits     | 8 | 16 | 24 | 32 | 48 | 64 | 96 | 128 
   ---        |   ---  |   ---   |   ---   |   ---   |   ---   |   ---   |   ---   |   ---   
  cifar10-gist@ALL  | 0.1131 | 0.1240  | 0.1264  | 0.1288  | 0.1377  | 0.1464  | 0.1482  | 0.1510
  cifar10-alexnet@ALL | 0.1339 | 0.1262 | 0.1471 | 0.1574 | 0.1769 | 0.1658 | 0.1854 | 0.1933
  nus-wide-tc21-alexnet@5000 | 0.4263 | 0.4827 | 0.4859 | 0.5026 | 0.5941 | 0.6221 | 0.6678 | 0.6940
  imagenet-tc100-alexnet@1000 | 0.0501 | 0.0700 | 0.0954 | 0.1258 | 0.1843 | 0.2308 | 0.3043 | 0.3416
