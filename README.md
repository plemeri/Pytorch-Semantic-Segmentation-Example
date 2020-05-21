# Pytorch-Semantic-Segmentation-Example
Example for semantic segmentation with pytorch.  
This repository is for [CSED514] Pattern Recognition on POSTECH.  

This project is based on [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) and [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105) on [Daimler Pedestrian Segmentation Benchmark](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/Daimler_Pedestrian_Segmentatio/daimler_pedestrian_segmentatio.html).

## Requirements
* python 3.x
* pytorch 1.2 or higher - GPU version recommended
* torchvision 0.4 or higher

## Dataset
Please download dataset from this [link](http://www.gavrila.net/data/Daimler/bmvc13-flohr-gavrila/PedCut2013_SegmentationDataset.tar.gz).  
Unzip the file and move `PedCut2013_SegmentationDataset` folder into the project folder.


## Train

DeepLab  

```sh
python train.py --model DeepLab --backbone ResNet50 --class_num 2 --stride 16 --batch_size 8 --learning_rate 0.01 --epochs 40 --device_ids 0 --checkpoint_dir ./checkpoint_deeplab --save_per_epoch 5
```

PspNet  

```sh
python train.py --model PspNet --backbone ResNet50 --class_num 2 --stride 8 --batch_size 8 --learning_rate 0.01 --epochs 40 --device_ids 0 --checkpoint_dir ./checkpoint_pspnet --save_per_epoch 5
```

Options:
- `--model` (str) - Choose from [DeepLab, PspNet].
- `--backbone` (str) - Choose from [ResNet18, ResNet50, Resnet101, ResNet152].
- `--class_num` (int) - Number of classes (for PennFudanPed, there are only background and foreground, so it's 2 classes).
- `--stride` (int) - Output stride of backbone network. Choose from [8, 16].
- `--batch_size` (int) - Mini batch size for training.
- `--learning_rate` (float) - Initial learning rate for training.
- `--epochs` (int) - Total number of epochs for training.
- `--device_ids` (int nargs) - GPU device ids for training. If there is more than one GPU, the model will be trained with multiple GPUs.
- `--checkpoint_dir` (str) - Checkpoint will be stored or loaded from this location.
- `--save_per_epoch` (int) - every K epoch the model will be saved.

## Test (Evaluation)

DeepLab  
```sh
python test.py --model DeepLab --backbone ResNet50 --class_num 2 --stride 16 --device_ids 0 --checkpoint_dir ./checkpoint_deeplab --result_dir ./result
```

PspNet  
```sh
python test.py --model PspNet --backbone ResNet50 --class_num 2 --stride 8 --device_ids 0 --checkpoint_dir ./checkpoint_pspnet --result_dir ./result
```

Options:
- `--model` (str) - Choose from [DeepLab, PspNet].
- `--backbone` (str) - Choose from [ResNet18, ResNet50, Resnet101, ResNet152].
- `--class_num` (int) - Number of classes (for PennFudanPed, there are only background and foreground, so it's 2 classes).
- `--stride` (int) - Output stride of backbone network. Choose from [8, 16].
- `--device_ids` (int nargs) - GPU device ids for evaluation.
- `--checkpoint_dir` (str) - Checkpoint will be loaded from this location.
- `--result_dir` (str) - every image, label, prediction will be saved to this location.
