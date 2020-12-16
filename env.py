import os 
import sys
import json
import torch
import torchvision
import numpy as np
import argparse
import torchvision.transforms as transforms
import cv2
from src.DDPG import Painter
from src.util import *
from PIL import Image
from torchvision import transforms, utils
import torchvision.datasets as datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

aug = transforms.Compose(
            [transforms.ToPILImage(),
            # transforms.Resize(128), # ONLY FOR MNIST !!!!
            transforms.RandomHorizontalFlip(),
             ])

width = 128 
convas_area = width * width

class Paint:
    def __init__(self, batch_size, max_step, k):
        self.batch_size = batch_size
        self.max_step = max_step
        self.k = k
        self.action_space = (13)
        self.observation_space = (self.batch_size, width, width, 3)  # for grayscale
        self.test = False

        self.img_train = []
        self.img_test = []
        self.train_num = 0
        self.test_num = 0

        self.painter = Painter('./renderer.pkl')
        self.painter.to(device)
    
    def load_chinese_img(self, train, dataset_size):
        path = "./data/chinese"
        char_file = ""
        if train:
            if dataset_size == "mini":
                char_file = "train_list_mini.txt"
            else:
                char_file = "train_list.txt"
        else:
            if dataset_size == "mini":
                char_file = "test_list_mini.txt"
            else:
                char_file = "test_list.txt"
        char_list = self.img_train if train else self.img_test
        with open(os.path.join(path, char_file), 'r') as f:
                chars = f.readlines()
        for c in chars:
            sub_folder = os.path.join(path, c[:-1])
            for img_path in os.listdir(sub_folder):
                if img_path.endswith(".png"):
                    try:
                        img = cv2.imread(os.path.join(sub_folder,img_path), 0)
                        img = cv2.resize(img, (width, width))
                        char_list.append(img)
                    except cv2.error:
                        print(f"cannot resize")

    def load_data(self, dataset, dataset_size):
        """
        @param dataset: A String representing the dataset
        """
        if dataset == "MNIST":
            self.img_train = datasets.MNIST(root='./data', train=True, download=True, transform=None).data
            self.img_test = datasets.MNIST(root='./data', train=False, download=True, transform=None).data
            self.train_num = len(self.img_train)
            self.test_num = len(self.img_test)
        elif dataset == "chinese": 
            self.load_chinese_img(True, dataset_size)
            self.load_chinese_img(False, dataset_size)
            self.train_num = len(self.img_train)
            self.test_num = len(self.img_test)
        else: # CelebA
            for i in range(200000):
                img_id = '%06d' % (i + 1)
                try:
                    img = cv2.imread('./data/img_align_celeba/' + img_id + '.jpg', cv2.IMREAD_UNCHANGED)
                    img = cv2.resize(img, (width, width))
                    if i > 2000:                
                        self.train_num += 1
                        self.img_train.append(img)
                    else:
                        self.test_num += 1
                        self.img_test.append(img)
                finally:
                    if (i + 1) % 10000 == 0:                    
                        print('loaded {} images'.format(i + 1))
        print('finish loading data, {} training images, {} testing images'.format(str(self.train_num), str(self.test_num)))
        
    def pre_data(self, id, test):
        if test:
            img = self.img_test[id]
        else:
            img = self.img_train[id]
        # sharpen the image
        img = np.where(img < 100, 0, 255).astype(np.uint8)
        img = aug(255 - img)
        img = np.asarray(img)
        return img  # for grayscale
    
    def reset(self, test=False, begin_num=False):
        self.test = test
        self.imgid = [0] * self.batch_size
        self.gt = torch.zeros([self.batch_size, 1, width, width], dtype=torch.uint8).to(device)  # for grayscale
        for i in range(self.batch_size):
            if test:
                id = (i + begin_num)  % self.test_num
            else:
                id = np.random.randint(self.train_num)
            self.imgid[i] = id
            self.gt[i] = torch.tensor(self.pre_data(id, test))
        self.tot_reward = ((self.gt.float() / 255) ** 2).mean(1).mean(1).mean(1)
        self.stepnum = 0
        self.canvas = torch.zeros([self.batch_size, 1, width, width], dtype=torch.uint8).to(device)  # for grayscale
        self.lastdis = self.ini_dis = self.cal_dis()
        return self.observation()
    
    def observation(self):
        # canvas B * 1 * width * width
        # gt B * 1 * width * width
        # T B * 1 * width * width
        ob = []
        T = torch.ones([self.batch_size, 1, width, width], dtype=torch.uint8) * self.stepnum
        return torch.cat((self.canvas, self.gt, T.to(device)), 1) # canvas, img, T

    def cal_trans(self, s, t):
        return (s.transpose(0, 3) * t).transpose(0, 3)
    
    def step(self, action):
        self.canvas = (self.painter.paint(action, self.canvas.float() / 255, self.k)[0] * 255).byte()
        self.stepnum += 1
        ob = self.observation()
        done = (self.stepnum == self.max_step)
        reward = self.cal_reward() # np.array([0.] * self.batch_size)
        return ob.detach(), reward, np.array([done] * self.batch_size), None

    def cal_dis(self):
        return (((self.canvas.float() - self.gt.float()) / 255) ** 2).mean(1).mean(1).mean(1)
    
    def cal_reward(self):
        dis = self.cal_dis()
        reward = (self.lastdis - dis) / (self.ini_dis + 1e-8)
        self.lastdis = dis
        return to_numpy(reward)