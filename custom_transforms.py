import torch
import numpy as np

from PIL import Image, ImageOps, ImageFilter, ImageEnhance


class RemoveID(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        label = np.array(sample['label'])
        label = (label != 0).astype(np.uint8)
        sample['label'] = Image.fromarray(label)

        return sample


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = np.array(sample['image']).astype(np.float32)
        image /= 255
        image -= self.mean
        image /= self.std

        sample['image'] = image

        return sample


class RandomGaussianBlur(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample['image']
        if np.random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=np.random.random()))
        sample['image'] = image

        return sample


class RandomHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        if np.random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        sample['image'] = image
        sample['label'] = label

        return sample


class RandomScaleRandomCrop(object):
    def __init__(self, base_size, crop_size, scale_range=(0.5, 2.0), ignore_mask=255):

        if '__iter__' not in dir(base_size):
            self.base_size = (base_size, base_size)
        else:
            self.base_size = base_size

        if '__iter__' not in dir(crop_size):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

        self.scale_range = scale_range
        self.ignore_mask = ignore_mask

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        width, height = image.size
        scale = np.random.rand() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]

        if width > height:
            resize_height = int(scale * self.base_size[1])
            resize_width = int(width * (resize_height / height))

        else:
            resize_width = int(scale * self.base_size[0])
            resize_height = int(height * (resize_width / width))

        image = image.resize((resize_width, resize_height), Image.BILINEAR)
        label = label.resize((resize_width, resize_height), Image.NEAREST)

        padding = [0, 0]
        if resize_width < self.crop_size[0]:
            padding[0] = self.crop_size[0] - resize_width

        if resize_height < self.crop_size[1]:
            padding[1] = self.crop_size[1] - resize_height

        if np.sum(padding) != 0:
            image = ImageOps.expand(image, (0, 0, *padding), fill=0)
            label = ImageOps.expand(label, (0, 0, *padding), fill=self.ignore_mask)

        width, height = image.size
        crop_coordinate = np.array([np.random.randint(0, width - self.crop_size[0] + 1),
                                    np.random.randint(0, height - self.crop_size[1] + 1)])

        image = image.crop((*crop_coordinate, *(crop_coordinate + self.crop_size)))
        label = label.crop((*crop_coordinate, *(crop_coordinate + self.crop_size)))

        sample['image'] = image
        sample['label'] = label

        return sample


class FixedSizedCenterCrop(object):
    def __init__(self, crop_size):
        if '__iter__' not in dir(crop_size):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        width, height = image.size

        if width > height:
            resize_height = int(self.crop_size[1])
            resize_width = int(width * (resize_height / height))

        else:
            resize_width = int(self.crop_size[0])
            resize_height = int(height * (resize_width / width))

        image = image.resize((resize_width, resize_height), Image.BILINEAR)
        label = label.resize((resize_width, resize_height), Image.NEAREST)

        crop_coordinate = np.array([int(resize_width - self.crop_size[0]) // 2,
                                    int(resize_height - self.crop_size[1]) // 2])

        image = image.crop((*crop_coordinate, *(crop_coordinate + self.crop_size)))
        label = label.crop((*crop_coordinate, *(crop_coordinate + self.crop_size)))

        sample['image'] = image
        sample['label'] = label

        return sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.array(sample['image']).astype(np.float32)
        label = np.array(sample['label'])
        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label)

        sample['image'] = image
        sample['label'] = label

        return sample