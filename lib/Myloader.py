import torch
from torch.utils.data import Dataset,DataLoader
import os
import cv2
import pandas as pd
import numpy as np
from torchvision import transforms as T, utils
from PIL import Image
import random
from matplotlib import pyplot as plt
from scipy.io import loadmat
import math
from lib.get_density_map import get_fix_gaussian,get_geo_gaussian
import scipy.misc as misc



class Shanghaitech(Dataset):
    def __init__(self,imdir,gtdir,transform = 0,train = True,test = False):
        self.imdir = imdir
        self.gtdir = gtdir
        self.train = train
        self.test = test
        self.transform = transform
        self.im = os.listdir(self.imdir)


    def __len__(self):

        return len(self.im)

    def __getitem__(self, idx):
        im_name = os.path.join(self.imdir,self.im[idx])
        gt_name = os.path.join(self.gtdir,self.im[idx].split('.')[0] + '.csv')


        # img = cv2.imread(im_name,0)
        img = misc.imread(im_name)
        img = img.astype(np.float32, copy=False)

        den = pd.read_csv(gt_name, sep=',',
                              header=None).as_matrix()
        den = den.astype(np.float32, copy=False)

        ht = img.shape[0]
        wd = img.shape[1]
        ht_1 = (ht / 4) * 4
        wd_1 = (wd / 4) * 4
        img = cv2.resize(img, (int(wd_1), int(ht_1)))
        # wd_1 = wd_1 / 4
        # ht_1 = ht_1 / 4
        den = cv2.resize(den, (int(wd_1), int(ht_1)))
        den = den * ((wd * ht) / (wd_1 * ht_1))



        if self.train:
            pro = random.random()
            if pro>=self.transform:
                    img = cv2.flip(img, 1)
                    den = cv2.flip(den, 1)
            self.den = den.reshape(1, den.shape[0], den.shape[1])
            self.img = img.reshape(1, img.shape[0], img.shape[1])
            return torch.Tensor(self.img),torch.Tensor(self.den)

        if self.test:

            self.img = img.reshape(1, img.shape[0], img.shape[1])
            self.den = den.reshape(1, den.shape[0], den.shape[1])

            return torch.Tensor(self.img), torch.Tensor(self.den)


class SHTech(Dataset):

    def __init__(self,imdir,gtdir,transform = 0,train = True,test = False,raw = False,num_cut = 4,geo = False):
        self.imdir = imdir
        self.gtdir = gtdir
        self.train = train
        self.test = test
        self.transform = transform
        self.imname = os.listdir(self.imdir)
        self.gtname = ['GT_'+name.replace('jpg', 'mat') for name in self.imname]
        self.imgs = []
        self.gts = []
        self.num_it = len(self.imname)
        self.num_cut = num_cut
        self.raw = raw
        print('Loading data,wait a second')
        PIXEL_MEANS = (0.485, 0.456, 0.406)
        PIXEL_STDS = (0.229, 0.224, 0.225)
        for idx in range(self.num_it):
            if idx %30 == 0:
                print ('loaded %d imgs'%idx)
            imname = os.path.join(self.imdir,self.imname[idx])


            img = cv2.imread(imname)
            img = img[:, :, ::-1]
            img = img.astype(np.float32, copy=False)
            img /= 255.0
            img -= np.array(PIXEL_MEANS)
            img /= np.array(PIXEL_STDS)
            gtname = os.path.join(self.gtdir, self.gtname[idx])
            image_info = loadmat(gtname)['image_info']
            annPoints = image_info[0][0][0][0][0] - 1

            if self.test or self.raw:
                h = img.shape[0]
                w = img.shape[1]
                c = img.shape[2]
                img = cv2.resize(img, (int(w / 8) * 8, int(h / 8) * 8), interpolation=cv2.INTER_LANCZOS4)
                if not geo:
                    den = get_fix_gaussian(img, annPoints)
                else:
                    den = get_geo_gaussian(img,annPoints)

                img = img.transpose([2, 0, 1])
                den = den.transpose([2,0,1])


                self.imgs.append(img)
                self.gts.append(den)

            if self.train and not self.raw:
                self.imgs.append(img)
                if not geo:
                    den = get_fix_gaussian(img, annPoints)
                else:
                    den = get_geo_gaussian(img,annPoints)
                self.gts.append(den)


    def __len__(self):

        return self.num_it

    def __getitem__(self, idx):


        img = self.imgs[idx]
        den = self.gts[idx]

        im_den = []
        im_sam = []
        enough = True



        if self.test or self.raw:

            return torch.Tensor(img),torch.Tensor(den)

        if self.train and not self.raw:

            h= img.shape[0]
            w = img.shape[1]
            c = img.shape[2]
            wn2, hn2 = w / 8, h / 8
            wn2, hn2 = int(wn2 / 8) * 8, int(hn2 / 8) * 8  # 1/8 to the original image size

            a_w, b_w = wn2 + 1, w - wn2  # 1/8+1,7/8
            a_h, b_h = hn2 + 1, h - hn2


            while enough:

                # for j in range(0, self.num_cut):
                    r1 = random.random()            #0<r1<1
                    r2 = random.random()
                    x = math.floor((b_w - a_w) * r1 + a_w) #choose the sample center randomly between (1/8 + r1*7/8) image size
                    y = math.floor((b_h - a_h) * r2 + a_h)
                    x1, y1 = int(x - wn2), int(y - hn2)         #sample offset
                    x2, y2 = int(x + wn2 - 1), int(y + hn2 - 1)

                    im_sampled = img[y1-1:y2, x1-1:x2,:]       #sample the image as the size of offset
                    im_density_sampled = den[y1-1:y2, x1-1:x2,:]           #sample the density map in the according area as the size of offset

                    pro = random.random()
                    if pro>=self.transform:
                            im_sampled = cv2.flip(im_sampled, 1)
                            im_density_sampled = cv2.flip(im_density_sampled, 1)


                    if np.sum(im_density_sampled) > 0:

                        im_density_sampled = im_density_sampled.transpose([2,0,1])
                        im_sampled = im_sampled.transpose([2, 0, 1])
                        im_density_sampled = im_density_sampled[np.newaxis,...]
                        im_sampled = im_sampled[np.newaxis,...]
                        im_den.append(torch.Tensor(im_density_sampled))
                        im_sam.append(torch.Tensor(im_sampled))

                    if len(im_den) >= self.num_cut:
                            enough = False



            self.img = torch.cat(im_sam,0)
            self.den = torch.cat(im_den,0)
            return self.img,self.den


class SDNSHTech(Dataset):

    def __init__(self,imdir,gtdir,transform = 0,train = True,test = False,raw = False,num_cut = 2,geo = False):
        self.imdir = imdir
        self.gtdir = gtdir
        self.train = train
        self.test = test
        self.transform = transform
        self.imname = os.listdir(self.imdir)
        self.gtname = ['GT_'+name.replace('jpg', 'mat') for name in self.imname]
        self.imgs = []
        self.gts = []
        self.num_it = len(self.imname)
        self.num_cut = num_cut
        self.raw = raw
        print('Loading data,wait a second')
        PIXEL_MEANS = (0.485, 0.456, 0.406)
        PIXEL_STDS = (0.229, 0.224, 0.225)
        for idx in range(self.num_it):
            if idx %30 == 0:
                print ('loaded %d imgs'%idx)
            imname = os.path.join(self.imdir,self.imname[idx])


            img = cv2.imread(imname)
            img = img[:, :, ::-1]
            img = img.astype(np.float32, copy=False)
            img /= 255.0
            img -= np.array(PIXEL_MEANS)
            img /= np.array(PIXEL_STDS)
            gtname = os.path.join(self.gtdir, self.gtname[idx])
            image_info = loadmat(gtname)['image_info']
            annPoints = image_info[0][0][0][0][0] - 1

            if self.test or self.raw:
                h = img.shape[0]
                w = img.shape[1]
                c = img.shape[2]
                img = cv2.resize(img, (int(w / 8) * 8, int(h / 8) * 8), interpolation=cv2.INTER_LANCZOS4)
                if not geo:
                    den = get_fix_gaussian(img, annPoints)
                else:
                    den = get_geo_gaussian(img,annPoints)

                img = img.transpose([2, 0, 1])
                den = den.transpose([2,0,1])


                self.imgs.append(img)
                self.gts.append(den)

            if self.train and not self.raw:
                self.imgs.append(img)
                if not geo:
                    den = get_fix_gaussian(img, annPoints)
                else:
                    den = get_geo_gaussian(img,annPoints)
                self.gts.append(den)


    def __len__(self):

        return self.num_it

    def __getitem__(self, idx):


        img = self.imgs[idx]
        den = self.gts[idx]

        im_den = []
        im_sam = []
        enough = True



        if self.test or self.raw:

            return torch.Tensor(img),torch.Tensor(den)

        if self.train and not self.raw:

            h= img.shape[0]
            w = img.shape[1]
            c = img.shape[2]
            # wn2, hn2 = w / 4, h / 4
            # wn2, hn2 = int(wn2 / 4) * 4, int(hn2 / 4) * 4  # 1/8 to the original image size
            wn2, hn2 = w / 4, h / 4
            wn2, hn2 = int(wn2 / 8) * 8, int(hn2 / 8) * 8  # 1/8 to the original image size


            a_w, b_w = wn2 + 1, w - wn2  # 1/8+1,7/8
            a_h, b_h = hn2 + 1, h - hn2


            while enough:

                # for j in range(0, self.num_cut):
                    r1 = random.random()            #0<r1<1
                    r2 = random.random()
                    x = math.floor((b_w - a_w) * r1 + a_w) #choose the sample center randomly between (1/8 + r1*7/8) image size
                    y = math.floor((b_h - a_h) * r2 + a_h)
                    x1, y1 = int(x - wn2), int(y - hn2)         #sample offset
                    x2, y2 = int(x + wn2 - 1), int(y + hn2 - 1)

                    im_sampled = img[y1-1:y2, x1-1:x2,:]       #sample the image as the size of offset
                    im_density_sampled = den[y1-1:y2, x1-1:x2,:]           #sample the density map in the according area as the size of offset

                    pro = random.random()
                    if pro>=self.transform:
                            im_sampled = cv2.flip(im_sampled, 1)
                            im_density_sampled = cv2.flip(im_density_sampled, 1)


                    if np.sum(im_density_sampled) > 0:

                        im_density_sampled = im_density_sampled.transpose([2,0,1])
                        im_sampled = im_sampled.transpose([2, 0, 1])
                        im_density_sampled = im_density_sampled[np.newaxis,...]
                        im_sampled = im_sampled[np.newaxis,...]
                        im_den.append(torch.Tensor(im_density_sampled))
                        im_sam.append(torch.Tensor(im_sampled))

                    if len(im_den) >= self.num_cut:
                            enough = False



            self.img = torch.cat(im_sam,0)
            self.den = torch.cat(im_den,0)
            return self.img,self.den








