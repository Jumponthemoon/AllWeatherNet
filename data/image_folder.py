###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def store_dataset_p2p(dir):
    A_paths = []
    A_images = []
    B_images = []
    B_paths = []
    A_file_names = os.listdir(dir+'/trainA')
    B_file_names = os.listdir(dir+'/trainB')
    for A_file in A_file_names:
        A_path = dir+'/trainA/'+A_file
        #print(A_path)
        A_paths.append(A_path)
        #print(A_path)
        A_images.append(Image.open(A_path).convert('RGB'))
        B_path = A_path[:-9]+'_ref'+A_path[-9::]
        B_path = B_path.replace('trainA','trainB')
        B_paths.append(B_path)
        B_images.append(Image.open(B_path).convert('RGB'))
        

    return A_images, A_paths,B_images, B_paths

def store_dataset(dir):
    images = []
    all_path = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                img = Image.open(path).convert('RGB')
                images.append(img)
                all_path.append(path)

    return images, all_path



def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
