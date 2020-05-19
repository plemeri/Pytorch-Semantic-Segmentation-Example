# Pytorch-Semantic-Segmentation-Example
Example for semantic segmentation with pytorch  
This project is based on [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) and [Daimler Pedestrian Segmentation Benchmark](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/Daimler_Pedestrian_Segmentatio/daimler_pedestrian_segmentatio.html).

## Requirements
* python 3.x
* pytorch 1.2 or higher - GPU version recommended
* torchvision 0.4 or higher

## Dataset
Please download dataset from this [link](http://www.gavrila.net/data/Daimler/bmvc13-flohr-gavrila/PedCut2013_SegmentationDataset.tar.gz).  
Unzip the file and move `PedCut2013_SegmentationDataset` folder into the project folder.


## Train

```sh
python train.py --backbone ResNet50 --class_num 2 --stride 16 --batch_size 8 --learning_rate 0.1 --epochs 40 --device_ids 0,1 --ignore_mask 255 --checkpoint_dir ./checkpoint --save_per_epoch 5
```

Options:
- `--backbone` (str) - Choose from [ResNet18, ResNet50, Resnet101, ResNet152].
- `--class_num` (int) - Number of classes (for PennFudanPed, there are only background and foreground, so it's 2 classes).
- `--stride` (int) - Output stride of backbone network. Choose from [8, 16].
- `--batch_size` (int) - Mini batch size for training.
- `--learning_rate` (float) - Initial learning rate for training.
- `--epochs` (int) - Total number of epochs for training.
- `--device_ids` (int nargs) - GPU device ids for training. If there is more than one GPU, the model will be trained with multiple GPUs.
- `--ignore_mask` (int) - Every pixel with ignore_mask value will be ignored for both training and testing. 
- `--checkpoint_dir` (str) - Checkpoint will be stored or loaded from this location. (
- `--save_per_epoch` (int) - every K epoch the model will be saved.

## Test (Evaluation)

```sh
python test.py --backbone ResNet50 --class_num 2 --stride 16 --device_ids 0,1 --ignore_mask 255 --checkpoint_dir ./checkpoint --result_dir ./result
```

Options:
- `--backbone` (str) - Choose from [ResNet18, ResNet50, Resnet101, ResNet152].
- `--class_num` (int) - Number of classes (for PennFudanPed, there are only background and foreground, so it's 2 classes).
- `--stride` (int) - Output stride of backbone network. Choose from [8, 16].
- `--ignore_mask` (int) - Every pixel with ignore_mask value will be ignored for both training and testing. 
- `--checkpoint_dir` (str) - Checkpoint will be loaded from this location.
- `--result_dir` (str) - every image, label, prediction will be saved to this location.
