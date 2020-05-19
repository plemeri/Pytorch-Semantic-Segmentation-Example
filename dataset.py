import os

from torch.utils.data import Dataset
from torchvision import transforms
from custom_transforms import *


class PedestrianSegmentation(Dataset):
    def __init__(self, dataset_dir='./PedCut2013_SegmentationDataset', split='train'):
        super(PedestrianSegmentation, self).__init__()
        self.dataset_dir = dataset_dir
        self.split = split

        self.image_dir = os.path.join(self.dataset_dir, 'data', 'completeData', 'left_images')
        self.label_dir = os.path.join(self.dataset_dir, 'data', 'completeData', 'left_groundTruth')

        self.image_list = os.listdir(self.image_dir)
        self.image_list.sort()

        self.label_list = os.listdir(self.label_dir)
        self.label_list.sort()

        if split == 'train':
            self.transform = transforms.Compose([
                RandomHorizontalFlip(),
                RandomScaleRandomCrop([128, 256], [128, 256], [0.8, 1.2], 255),
                RandomGaussianBlur(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                FixedSizedCenterCrop([128, 256]),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensor()
            ])

    def __getitem__(self, index):
        image_item = self.image_list[index]
        label_item = self.label_list[index]

        image = Image.open(os.path.join(self.image_dir, image_item))
        label = Image.open(os.path.join(self.label_dir, label_item))
        label = Image.fromarray(np.array(label).astype(np.uint8))

        sample = {'image': image, 'label': label}
        sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_list)
    

class Metrics:
    def __init__(self, class_num, ignore_mask):
        self.class_num = class_num
        self.ignore_mask = ignore_mask
        self.confusion_matrix = np.zeros((self.class_num, self.class_num))
        self.class_iou = []
        self.mean_iou = 0.0
        self.accuracy = 0.0

    def __call__(self, prediction, label, mode='train'):
        self.compute_confusion_matrix_and_add_up(label, prediction)
        if mode == 'train':
            accuracy = self.compute_pixel_accuracy()
            metric_dict = {'accuracy': accuracy}
        else:
            class_iou = self.compute_class_iou()
            mean_iou = self.compute_mean_iou()
            accuracy = self.compute_pixel_accuracy()
            metric_dict = {'mean_iou': mean_iou, 'accuracy': accuracy}
            for i, iou in enumerate(class_iou):
                metric_dict['class_iou_' + str(i)] = iou
        return metric_dict

    def clear_confusion_matrix(self):
        self.confusion_matrix = np.zeros((self.class_num, self.class_num))

    def compute_confusion_matrix(self, label, image):
        if len(label.shape) == 4:
            label = torch.argmax(label, dim=1)
        if len(image.shape) == 4:
            image = torch.argmax(image, dim=1)

        label = label.flatten().cpu().numpy().astype(np.int64)
        image = image.flatten().cpu().numpy().astype(np.int64)

        valid_indices = (label != self.ignore_mask) & (0 <= label) & (label < self.class_num)

        enhanced_label = self.class_num * label[valid_indices].astype(np.int32) + image[valid_indices]
        confusion_matrix = np.bincount(enhanced_label, minlength=self.class_num * self.class_num)
        confusion_matrix = np.reshape(confusion_matrix, (self.class_num, self.class_num))

        return confusion_matrix

    def compute_confusion_matrix_and_add_up(self, label, image):
        self.confusion_matrix += self.compute_confusion_matrix(label, image)

    def compute_pixel_accuracy(self):
        return np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)

    def compute_class_iou(self):
        class_iou = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=0) + np.sum(self.confusion_matrix, axis=1) - np.diag(
                self.confusion_matrix))
        return class_iou

    def compute_mean_iou(self):
        class_iou = self.compute_class_iou()
        return np.nanmean(class_iou)
