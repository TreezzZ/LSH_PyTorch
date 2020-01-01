# Similarity Search in High Dimensions via Hashing

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
  cifar10-gist@ALL  | 0.1138 | 0.1191  | 0.1195  | 0.1288  | 0.1349  | 0.1436  | 0.1536  | 0.1521
  cifar10-alexnet@ALL | 0.1463 | 0.1290 | 0.1416 | 0.1588 | 0.1686 | 0.1757 | 0.1860 | 0.2121
  nus-wide-tc21-alexnet@5000 | 0.3905 | 0.4632 | 0.4836 | 0.5243 | 0.6012 | 0.6051 | 0.6513 | 0.6921
  imagenet-tc100-alexnet@1000 | 0.0536 | 0.0685 | 0.0952 | 0.1290 | 0.1861 | 0.2326 | 0.2909 | 0.3410
