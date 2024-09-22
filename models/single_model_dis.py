import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
import random
#from . import depth_networks
from . import networks
import sys
import torchvision.transforms as transforms
# from .Myloss import *

tensor = torch.randn(8, 3, 16, 16)

def select_crop_center(probability_map, crop_size=30):


    flattened_probs = probability_map.flatten()


    center_idx = torch.multinomial(flattened_probs / flattened_probs.sum(), 1).item()


    center_y = center_idx // probability_map.shape[1]
    center_x = center_idx % probability_map.shape[1]
    top_left_x = max(center_x - crop_size // 2, 0)
    top_left_y = max(center_y - crop_size // 2, 0)
    return top_left_x, top_left_y

class SingleModel(BaseModel):
    def name(self):
        return 'SingleGANModel'

    def initialize(self, opt):

        ################################################################
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A_gray = self.Tensor(nb, 1, size, size)
        self.center_crop = transforms.CenterCrop(size//16) 
        self.l1_loss = nn.L1Loss()


        if opt.vgg > 0:
            self.vgg_loss = networks.PerceptualLoss(opt)
            if self.opt.IN_vgg:
                self.vgg_patch_loss = networks.PerceptualLoss(opt)
                self.vgg_patch_loss.cuda()
            self.vgg_loss.cuda()
            self.vgg = networks.load_vgg16("./model", self.gpu_ids)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif opt.fcn > 0:
            self.fcn_loss = networks.SemanticLoss(opt)
            self.fcn_loss.cuda()
            self.fcn = networks.load_fcn("./model")
            self.fcn.eval()
            for param in self.fcn.parameters():
                param.requires_grad = False


        skip = True if opt.skip > 0 else False
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=skip, opt=opt)



        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, False)


            if self.opt.patchD:
                self.netD_P = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_patchD, opt.norm, use_sigmoid, self.gpu_ids, True)


        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            # self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                if self.opt.patchD:
                    self.load_network(self.netD_P, 'D_P', which_epoch)
        #import horovod.torch as hvd
        if self.isTrain:
            self.old_lr = opt.lr
            # self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            if opt.use_wgan:
                self.criterionGAN = networks.DiscLossWGANGP()
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            if opt.use_mse:
                self.criterionCycle = torch.nn.MSELoss()
            else:
                self.criterionCycle = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if self.opt.patchD:
                self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
                                                      )
            # self.optimizer_G =  hvd.DistributedOptimizer(self.optimizer_G, named_parameters=self.netG_A.named_parameters())
            # self.optimizer_D_A = hvd.DistributedOptimizer(self.optimizer_D_A, named_parameters=self.netD_A.named_parameters())
            # self.optimizer_D_P = hvd.DistributedOptimizer(self.optimizer_D_P, named_parameters=self.netD_P.named_parameters())
        
        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        # networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            if self.opt.patchD:
                networks.print_network(self.netD_P)
            # networks.print_network(self.netD_B)
        if opt.isTrain:
            # hvd.broadcast_parameters(self.netG_A.state_dict(), root_rank=0)
            # hvd.broadcast_parameters(self.netD_P.state_dict(), root_rank=0)
            # hvd.broadcast_parameters(self.netD_A.state_dict(), root_rank=0)

            self.netG_A.train()
            # self.netG_B.train()
        else:
            self.netG_A.eval()
            # self.netG_B.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_img = input['input_img']
        input_A_gray = input['A_gray']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A_gray.resize_(input_A_gray.size()).copy_(input_A_gray)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    


    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_gray = Variable(self.input_A_gray, volatile=True)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        # print(np.transpose(self.real_A.data[0].cpu().float().numpy(),(1,2,0))[:2][:2][:])
        if self.opt.skip == 1:
                self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray)

        else:
            self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray)
        # self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)


    def predict(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_gray = Variable(self.input_A_gray, volatile=True)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        # print(np.transpose(self.real_A.data[0].cpu().float().numpy(),(1,2,0))[:2][:2][:])
        
        if self.opt.skip == 1:
                self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray)
        else:
            self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray)
        # self.rec_A = self.netG_B.forward(self.fake_B)

        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)

        A_gray = util.atten2im(self.real_A_gray.data)
        # rec_A = util.tensor2im(self.rec_A.data)
        # if self.opt.skip == 1:
        #     latent_real_A = util.tensor2im(self.latent_real_A.data)
        #     latent_show = util.latent2im(self.latent_real_A.data)
        #     max_image = util.max2im(self.fake_B.data, self.latent_real_A.data)
        #     return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
        #                     ('latent_show', latent_show), ('max_image', max_image), ('A_gray', A_gray)])
        # else:
        #     return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])
        # return OrderedDict([('fake_B', fake_B)])
            #fake_B_latent = util.fake_latent2im(self.fake_B.data,self.latent_real_A.data)


        return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake, use_ragan):
        # Real
        pred_real = netD.forward(real)
        pred_fake = netD.forward(fake.detach())
        if self.opt.use_wgan:
            loss_D_real = pred_real.mean()
            loss_D_fake = pred_fake.mean()
            loss_D = loss_D_fake - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD, 
                                                real.data, fake.data)
        elif self.opt.use_ragan and use_ragan:
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B = self.fake_B
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, True)
        self.loss_D_A.backward()
    
    def backward_D_P(self):
        loss_D_P_low = 0
        if self.opt.hybrid_loss:
            loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch, self.fake_patch, False)
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P += self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i], False)
                    loss_D_P_low += self.backward_D_basic(self.netD_P, self.real_patch_1_low[i], self.fake_patch_1_low[i], False)

                self.loss_D_P = loss_D_P/float(self.opt.patchD_3 + 1) + loss_D_P_low/float(self.opt.patchD_3)


        if self.opt.D_P_times2:
            self.loss_D_P = self.loss_D_P*2
        self.loss_D_P.backward()

    # def backward_D_B(self):
    #     fake_A = self.fake_A_pool.query(self.fake_A)
    #     self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_A_gray = Variable(self.input_A_gray)
        self.real_img = Variable(self.input_img)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A))/(torch.max(self.real_A) - torch.min(self.real_A))
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_img, self.real_A_gray)
        else:
            self.fake_B = self.netG_A.forward(self.real_img, self.real_A_gray)
        # fake_B_features = self.encoder(self.fake_B, 'day', 'val')
        # self.fake_B_depth = self.depth_decoder(fake_B_features)[("disp", 0)]
        # real_A_features = self.encoder(self.real_A, 'day', 'val')
        # self.real_A_depth = self.depth_decoder(real_A_features)[("disp", 0)]


        if self.opt.patchD:

            w = self.real_A.size(3)
            h = self.real_A.size(2)

            # top_left_x, top_left_y = select_crop_center(self.real_A_gray)
            # w_offset = min(top_left_x,w - self.opt.patchSize - 1)
            # h_offset = min(top_left_y,h - self.opt.patchSize - 1)

            # if top_left_x + self.opt.patchSize+1 > w :
            #     center_x = w - self.opt.patchSize - 1
            # if center_y + self.opt.patchSize+1 > h :
            #     center_y = w - self.opt.patchSize - 1

            w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))

            self.fake_patch = self.fake_B[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]
            self.real_patch = self.real_B[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]
            self.input_patch = self.real_A[:,:, h_offset:h_offset + self.opt.patchSize,
                   w_offset:w_offset + self.opt.patchSize]
        if self.opt.patchD_3 > 0:
            self.fake_patch_1 = []
            self.real_patch_1 = []
            self.input_patch_1 = []
            self.fake_patch_1_low = []
            self.real_patch_1_low = []
            self.input_patch_1_low = []
            w = self.real_A.size(3)
            h = self.real_A.size(2)
            for i in range(self.opt.patchD_3):
                w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
                h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
                if self.opt.patch_window:
                    input_patch_batches = []
                    fake_patch_batches = []
                    real_patch_batches = []

                    for idx in range(self.opt.batchSize):
                        real_patch = self.real_B[idx,:, h_offset_1:h_offset_1 + self.opt.patchSize, w_offset_1:w_offset_1 + self.opt.patchSize]  
                        stride = 1
                        loss_item = []
                        input_patch_queue = []
                        fake_patch_queue = []

                        selected_offset = (0, 0)
                        min_loss = float('inf')

                        for x in [-3, 0,3]:
                            for y in [-3,0,3]:
                                temp_h_offset = max(0, min(h - self.opt.patchSize, h_offset_1 + x))
                                temp_w_offset = max(0, min(w - self.opt.patchSize, w_offset_1 + y))

                                input_patch = self.real_A[idx, :, temp_h_offset:temp_h_offset + self.opt.patchSize, temp_w_offset:temp_w_offset + self.opt.patchSize]
                                current_loss = self.l1_loss(input_patch, real_patch).item()
                                if current_loss < min_loss:
                                    min_loss = current_loss
                                    selected_offset = (temp_h_offset, temp_w_offset)

                        # Using the selected offset to crop the patches
                        selected_h_offset, selected_w_offset = selected_offset
                        input_patch = self.real_A[idx, :, selected_h_offset:selected_h_offset + self.opt.patchSize, selected_w_offset:selected_w_offset + self.opt.patchSize]
                        fake_patch = self.fake_B[idx, :, selected_h_offset:selected_h_offset + self.opt.patchSize, selected_w_offset:selected_w_offset + self.opt.patchSize]

                        input_patch_batches.append(input_patch)
                        fake_patch_batches.append(fake_patch)
                        real_patch_batches.append(real_patch)

                    fake_patch = torch.stack(fake_patch_batches, dim=0)
                    input_patch = torch.stack(input_patch_batches, dim=0)
                    real_patch = torch.stack(real_patch_batches, dim=0)

                    fake_patch_low = self.center_crop(fake_patch)
                    real_patch_low = self.center_crop(real_patch)
                    input_patch_low = self.center_crop(input_patch)
                    
                    self.fake_patch_1.append(fake_patch)
                    self.real_patch_1.append(real_patch)
                    self.input_patch_1.append(input_patch)

                    self.fake_patch_1_low.append(fake_patch_low)
                    self.real_patch_1_low.append(real_patch_low)
                    self.input_patch_1_low.append(input_patch_low)

                else:
                    fake_patch = self.fake_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                        w_offset_1:w_offset_1 + self.opt.patchSize]
                    real_patch = self.real_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                        w_offset_1:w_offset_1 + self.opt.patchSize] 
                    input_patch = self.real_A[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                        w_offset_1:w_offset_1 + self.opt.patchSize]  
    
                    fake_patch_low = self.center_crop(fake_patch)
                    real_patch_low = self.center_crop(real_patch)
                    input_patch_low = self.center_crop(input_patch)


                    self.fake_patch_1.append(fake_patch)
                    self.real_patch_1.append(real_patch)
                    self.input_patch_1.append(input_patch)

                    self.fake_patch_1_low.append(fake_patch_low)
                    self.real_patch_1_low.append(real_patch_low)
                    self.input_patch_1_low.append(input_patch_low)


    def backward_G(self, epoch):
        pred_fake = self.netD_A.forward(self.fake_B)
        if self.opt.use_wgan:
            self.loss_G_A = -pred_fake.mean()
        elif self.opt.use_ragan:
            pred_real = self.netD_A.forward(self.real_B)

            self.loss_G_A = (self.criterionGAN(pred_real - torch.mean(pred_fake), False) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), True)) / 2
            

            
        else:
            self.loss_G_A = self.criterionGAN(pred_fake, True)
        
        loss_G_A = 0
        loss_G_A_low = 0
        if self.opt.patchD:
            pred_fake_patch = self.netD_P.forward(self.fake_patch)
            if self.opt.hybrid_loss:
                loss_G_A += self.criterionGAN(pred_fake_patch, True)
            else:
                pred_real_patch = self.netD_P.forward(self.real_patch)
                
                loss_G_A += (self.criterionGAN(pred_real_patch - torch.mean(pred_fake_patch), False) +
                                      self.criterionGAN(pred_fake_patch - torch.mean(pred_real_patch), True)) / 2
        if self.opt.patchD_3 > 0:   
            for i in range(self.opt.patchD_3):
                pred_fake_patch_1 = self.netD_P.forward(self.fake_patch_1[i])

                if self.opt.hybrid_loss:
                    loss_G_A += self.criterionGAN(pred_fake_patch_1, True)
            if not self.opt.D_P_times2:
                self.loss_G_A += loss_G_A/float(self.opt.patchD_3 + 1)

        if self.opt.patchD_3 > 0:   
            for i in range(self.opt.patchD_3):
                pred_fake_patch_1_low = self.netD_P.forward(self.fake_patch_1_low[i])

                if self.opt.hybrid_loss:
                    loss_G_A_low += self.criterionGAN(pred_fake_patch_1_low, True)
                    
            if not self.opt.D_P_times2:
                self.loss_G_A += loss_G_A_low/float(self.opt.patchD_3)
                
        if epoch < 0:
            vgg_w = 0
        else:
            vgg_w = 1
        if self.opt.vgg > 0:
            self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg, 
                    self.fake_B, self.real_A) * self.opt.vgg if self.opt.vgg > 0 else 0
            if self.opt.patch_vgg:
                if not self.opt.IN_vgg:
                    loss_vgg_patch = self.vgg_loss.compute_vgg_loss(self.vgg, 
                    self.fake_patch, self.input_patch) * self.opt.vgg
                    loss_vgg_patch_low = 0
                else:
                    loss_vgg_patch = self.vgg_patch_loss.compute_vgg_loss(self.vgg, 
                    self.fake_patch, self.input_patch) * self.opt.vgg
                if self.opt.patchD_3 > 0:
                    for i in range(self.opt.patchD_3):
                        if not self.opt.IN_vgg:
                            loss_vgg_patch += self.vgg_loss.compute_vgg_loss(self.vgg, 
                                self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                            loss_vgg_patch_low += self.vgg_loss.compute_vgg_loss(self.vgg, 
                                self.fake_patch_1_low[i], self.input_patch_1_low[i]) * self.opt.vgg
                        else:
                            loss_vgg_patch += self.vgg_patch_loss.compute_vgg_loss(self.vgg, 
                                self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                        
                    self.loss_vgg_b += loss_vgg_patch/float(self.opt.patchD_3 + 1)
                    self.loss_vgg_b += loss_vgg_patch_low/float(self.opt.patchD_3)

                else:
                    self.loss_vgg_b += loss_vgg_patch
            # loss_TV = 20*self.L_TV(self.fake_B)
            # # #loss_spa = 5*torch.mean(self.L_spa(self.fake_B, self.real_A))
            # # #loss_col = 50*torch.mean(self.L_color(self.fake_B))
            # loss_exp = 10*torch.mean(self.L_exp(self.fake_B))        
            # dce_loss =  loss_TV + loss_exp
            #self.depth_loss = self.l1_loss(self.fake_B_depth,self.real_A_depth)
            # loss = F.l1_loss(predicted, target)


            self.loss_G = self.loss_G_A+ self.loss_vgg_b*vgg_w # +  self.depth_loss


        # self.loss_G = self.L1_AB + self.L1_BA
        self.loss_G.backward()


    # def optimize_parameters(self, epoch):
    #     # forward
    #     self.forward()
    #     # G_A and G_B
    #     self.optimizer_G.zero_grad()
    #     self.backward_G(epoch)
    #     self.optimizer_G.step()
    #     # D_A
    #     self.optimizer_D_A.zero_grad()
    #     self.backward_D_A()
    #     self.optimizer_D_A.step()
    #     if self.opt.patchD:
    #         self.forward()
    #         self.optimizer_D_P.zero_grad()
    #         self.backward_D_P()
    #         self.optimizer_D_P.step()
        # D_B
        # self.optimizer_D_B.zero_grad()
        # self.backward_D_B()
        # self.optimizer_D_B.step()
    def optimize_parameters(self, epoch):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G(epoch)
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        if not self.opt.patchD:
            self.optimizer_D_A.step()
        else:
            # self.forward()
            self.optimizer_D_P.zero_grad()
            self.backward_D_P()
            torch.nn.utils.clip_grad_norm_(self.netD_P.parameters(), max_norm=1.0)

            self.optimizer_D_A.step()
            self.optimizer_D_P.step()

        # with torch.no_grad():
        #             self.netG_A.module.scale_factor.clamp_(min=0.0, max=1.0)
    def get_current_errors(self, epoch):
        D_A = self.loss_D_A.item()
        D_P = self.loss_D_P.item() if self.opt.patchD else 0
        G_A = self.loss_G_A.item()
        if self.opt.vgg > 0:
            vgg = self.loss_vgg_b.item()/self.opt.vgg if self.opt.vgg > 0 else 0
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ("vgg", vgg), ("D_P", D_P)])
        elif self.opt.fcn > 0:
            fcn = self.loss_fcn_b.item()/self.opt.fcn if self.opt.fcn > 0 else 0
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ("fcn", fcn), ("D_P", D_P)])
        



    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        if self.opt.skip > 0:
            latent_real_A = util.tensor2im(self.latent_real_A.data)
            latent_show = util.latent2im(self.latent_real_A.data)
            if self.opt.patchD:
                fake_patch = util.tensor2im(self.fake_patch.data)
                real_patch = util.tensor2im(self.real_patch.data)
                fake_patch_low = util.tensor2im(self.fake_patch_1_low[0].data)
                real_patch_low = util.tensor2im(self.real_patch_1_low[0].data)
                fake_patch1 = util.tensor2im(self.fake_patch_1[1].data)
                real_patch1 = util.tensor2im(self.real_patch_1[1].data)

                fake_patch2 = util.tensor2im(self.fake_patch_1[2].data)
                real_patch2 = util.tensor2im(self.real_patch_1[2].data)

                fake_patch3 = util.tensor2im(self.fake_patch_1[3].data)
                real_patch3 = util.tensor2im(self.real_patch_1[3].data)

                fake_patch4 = util.tensor2im(self.fake_patch_1[4].data)
                real_patch4 = util.tensor2im(self.real_patch_1[4].data)
                if self.opt.patch_vgg:
                    input_patch = util.tensor2im(self.input_patch_1[0].data)
                    input_patch_low = util.tensor2im(self.input_patch_1_low[0].data)

                    if not self.opt.self_attention:
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), ('real_patch', real_patch),
                                ('fake_patch', fake_patch), ('input_patch', input_patch)])
                    else:
                        self_attention = util.atten2im(self.real_A_gray.data)
                        return OrderedDict(
                               [ ('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                ('latent_show', latent_show), ('real_B', real_B), 
                                ('real_patch', real_patch), ('fake_patch', fake_patch), 
                                ('real_patch1', real_patch1), ('fake_patch1', fake_patch1), 
                                ('real_patch2', real_patch2), ('fake_patch2', fake_patch2), 
                                ('real_patch3', real_patch3), ('fake_patch3', fake_patch3), 
                                ('real_patch4', real_patch4), ('fake_patch4', fake_patch4),
                                ('self_attention', self_attention)
                                ,('real_patch_low', real_patch_low),('fake_patch_low', fake_patch_low),('input_patch_low', input_patch_low) ])   

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        if self.opt.patchD:
            self.save_network(self.netD_P, 'D_P', label, self.gpu_ids)
        # self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        # self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def update_learning_rate(self):
        
        if self.opt.new_lr:
            lr = self.old_lr/2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        if self.opt.patchD:
            for param_group in self.optimizer_D_P.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
