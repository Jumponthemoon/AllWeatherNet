import torch
from torch import nn
import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, store_dataset, store_dataset_p2p
from PIL import Image
import PIL
from pdb import set_trace as st
import numpy as np
import torch.nn.functional as F

def pad_tensor(input):
    
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

            padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
            input = padding(input).data
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.shape[2], input.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:,:, pad_top: height - pad_bottom, pad_left: width - pad_right]

class UnalignedTestDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot)
        self.dir_B = '/home/chenghao/Project/dataset/dark_zurich/Dark_Zurich_train_anon/rgb_anon/train/night'

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)
        # self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A_size = A_img.size
        B_size = B_img.size
        A_size = A_size = (A_size[0]//16*16, A_size[1]//16*16)
        B_size = B_size = (B_size[0]//16*16, B_size[1]//16*16)
        A_img = A_img.resize(A_size, Image.BICUBIC)
        B_img = B_img.resize(B_size, Image.BICUBIC)


        A_img = self.transform(A_img)
        B_img = self.transform(B_img)
  
        if self.opt.resize_or_crop == 'no':
            pass
        else:
            w = A_img.size(2)
            h = A_img.size(1)
            size = [8,16,22]
            from random import randint
            size_index = randint(0,2)
            Cropsize = size[size_index]*16

            w_offset = random.randint(0, max(0, w - Cropsize - 1))
            h_offset = random.randint(0, max(0, h - Cropsize - 1))

            A_img = A_img[:, h_offset:h_offset + Cropsize,
                   w_offset:w_offset + Cropsize]

            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(2, idx)
                B_img = B_img.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(1, idx)
                B_img = B_img.index_select(1, idx)

        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
import albumentations as A
class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        #self.opt.resize_or_crop == 'no'
        # self.A_paths = make_dataset(self.dir_A)
        # self.B_paths = make_dataset(self.dir_B)
        #images, all_path
        #data_list = [i_id.strip() for i_id in open('/home/chenghao/Project/code/DANNet/dataset/lists/zurich_dn_pair_train.csv')]
        if opt.phase == 'train':

           self.A_imgs, self.A_paths, self.B_imgs, self.B_paths = store_dataset_p2p(self.root)
        else:
           self.A_imgs, self.A_paths = store_dataset(self.dir_A)
           self.B_imgs, self.B_paths = store_dataset(self.dir_B)
        #self.A_imgs, self.A_paths, self.B_imgs, self.B_paths = store_dataset3(self.root)
        # self.A_paths = sorted(self.A_paths)
        # self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_imgs)
        self.B_size = len(self.B_imgs)
        #self.opt.resize_or_crop = 'no'
        self.transform = get_transform(opt)
        # self.transform_A = A.Compose([
        #             A.HorizontalFlip(p=0.5),
        #         ])


    def __getitem__(self, index):
        # A_path = self.A_paths[index % self.A_size]
        # B_path = self.B_paths[index % self.B_size]

        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        A_img = self.A_imgs[index % self.A_size]
        B_img = self.B_imgs[index % self.B_size]
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        #print(A_path,B_path)
        # A_size = A_img.size
        # B_size = B_img.size
        # A_size = A_size = (A_size[0]//16*16, A_size[1]//16*16)
        # B_size = B_size = (B_size[0]//16*16, B_size[1]//16*16)
        # A_img = A_img.resize(A_size, Image.BICUBIC)
        # B_img = B_img.resize(B_size, Image.BICUBIC)
        # A_gray = A_img.convert('LA')
        # A_gray = 255.0-A_gray
        #print(A_img,type(A_img))
        seed = np.random.randint(0, 1000000)
        
        # # albumentation transform
        # random.seed(seed)
        # A_img = self.transform_A(image=np.array(A_img))['image']
        # random.seed(seed)
        # B_img = self.transform_A(image=np.array(B_img))['image']

        #pytorch transform
        torch.manual_seed(seed)
        A_img = self.transform(A_img)
        torch.manual_seed(seed)
        B_img = self.transform(B_img)


        self.opt.resize_or_crop = 'no'
        if self.opt.resize_or_crop == 'no':
            r,g,b = A_img[0]+1, A_img[1]+1, A_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
           # A_gray = 1 / (1 + np.exp(-5 * (A_gray - 0.5)))

            A_gray = -A_gray*(A_gray-2)
            #A_gray = 1. - (0.333*r+0.333*g+0.333*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
            #A_gray = F.tanh(torch.unsqueeze(A_gray, 0))
            input_img = A_img
            # A_gray = (1./A_gray)/255.
        else:
            w = A_img.size(2)
            h = A_img.size(1)
            
            # A_gray = (1./A_gray)/255.
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(2, idx)
                B_img = B_img.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(1, idx)
                B_img = B_img.index_select(1, idx)
            if self.opt.vary == 1 and (not self.opt.no_flip) and random.random() < 0.5:
                times = random.randint(self.opt.low_times,self.opt.high_times)/100.
                input_img = (A_img+1)/2./times
                input_img = input_img*2-1
            else:
                input_img = A_img
            if self.opt.lighten:
                B_img = (B_img + 1)/2.
                B_img = (B_img - torch.min(B_img))/(torch.max(B_img) - torch.min(B_img))
                B_img = B_img*2. -1
            r,g,b = input_img[0]+1, input_img[1]+1, input_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            #A_gray = 1. - (0.333*r+0.333*g+0.333*b)/2.

            A_gray = torch.unsqueeze(A_gray, 0)
        return {'A': A_img, 'B': B_img, 'A_gray': A_gray, 'input_img': input_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'


