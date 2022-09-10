import os
import torch
import torchvision
import random
import numpy as np
import imgaug.augmenters as iaa
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
aug_seq_source = iaa.Sequential([
    iaa.Sometimes(
        0.5,
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))
    ),
    iaa.Sometimes(
        0.2,
        iaa.Dropout(p=(0.01, 0.1))
    ),
    iaa.Sometimes(
        0.5,
        iaa.JpegCompression(compression=(10, 30))
    ),
])
aug_seq_all = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
    iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))
])

def transform_augment(img_list, split='val', min_max=(0, 1)):    
    if split == 'train':
        aug_seq_all_det = aug_seq_all.to_deterministic()
        img_list = [aug_seq_all_det(image = np.array(img)) for img in img_list]
        img_list[0] = aug_seq_source(image = np.array(img_list[0]))
        img_list = [Image.fromarray(img) for img in img_list]
    imgs = [totensor(img) for img in img_list]
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img