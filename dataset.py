import os

from torch.utils.data import Dataset
from torchvision import transforms
from custom_transforms import *


class PennFudanSegmentation(Dataset):
    def __init__(self, dataset_dir='./PennFudanPed', split='train'):
        super(PennFudanSegmentation, self).__init__()
        self.dataset_dir = dataset_dir
        self.split = split

        self.image_dir = os.path.join(self.dataset_dir, 'PNGImages')
        self.label_dir = os.path.join(self.dataset_dir, 'PedMasks')

        if os.path.isfile(os.path.join(self.dataset_dir, split + '.txt')) is False:
            train_list = open(os.path.join(self.dataset_dir, 'train.txt'), 'w')
            test_list = open(os.path.join(self.dataset_dir, 'test.txt'), 'w')

            files = [os.path.splitext(file)[0] for file in os.listdir(self.image_dir)]
            file_idx = np.linspace(0, len(files) - 1, len(files), dtype=int)
            test_file_idx = np.random.choice(file_idx, len(files) // 5, replace=False)
            train_file_idx = np.delete(file_idx, test_file_idx)

            for i in test_file_idx:
                test_list.writelines(files[i] + '\n')

            for i in train_file_idx:
                train_list.writelines(files[i] + '\n')

            test_list.close()
            train_list.close()

        self.data_list = open(os.path.join(self.dataset_dir, split + '.txt'), 'r').read().split('\n')
        self.data_list = [file for file in self.data_list if len(file) > 0]

        self.image_list = [file + '.png' for file in self.data_list]
        self.image_list.sort()

        self.label_list = [file + '_mask.png' for file in self.data_list]
        self.label_list.sort()

        if split == 'train':
            self.transform = transforms.Compose([
                RemoveID(),
                RandomHorizontalFlip(),
                RandomScaleRandomCrop([512, 512], [512, 512], [0.8, 1.2], 255),
                RandomGaussianBlur(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                RemoveID(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensor()
            ])

    def __getitem__(self, index):
        image_item = self.image_list[index]
        label_item = self.label_list[index]

        image = Image.open(os.path.join(self.image_dir, image_item))
        label = Image.open(os.path.join(self.label_dir, label_item))

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
