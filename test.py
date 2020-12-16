import os
import cv2
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from src.ResNet import *
from src.DDPG import Painter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
width = 128
ACTION_SPACE = 13

aug = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
             ])

parser = argparse.ArgumentParser(description='Learning to Paint')
parser.add_argument('--max_step', default=10, type=int, help='max length for episode')
parser.add_argument('--k', default=3, type=int, help='number of strokes to generate at each step')
parser.add_argument('--actor', default='./model/Paint-run1/actor.pkl', type=str, help='Actor model')
parser.add_argument('--renderer', default='./renderer.pkl', type=str, help='renderer model')
parser.add_argument('--img', default='image/test.png', type=str, help='test image')
parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
parser.add_argument('--divide', default=1, type=int, help='divide the target image to get better resolution')
args = parser.parse_args()

canvas_cnt = args.divide * args.divide
T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)
img = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
origin_shape = (img.shape[1], img.shape[0])
img = cv2.bitwise_not(img)

Painter = Painter(args.renderer)

def small2large(x):
    # (d * d, width, width) -> (d * width, d * width)    
    x = x.reshape(args.divide, args.divide, width, width, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(args.divide * width, args.divide * width, -1)
    return x

def large2small(x):
    # (d * width, d * width) -> (d * d, width, width)
    x = x.reshape(args.divide, width, args.divide, width, 1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(canvas_cnt, width, width, 1)
    return x

def smooth(img):
    def smooth_pix(img, tx, ty):
        if tx == args.divide * width - 1 or ty == args.divide * width - 1 or tx == 0 or ty == 0: 
            return img
        img[tx, ty] = (img[tx, ty] + img[tx + 1, ty] + img[tx, ty + 1] + img[tx - 1, ty] + img[tx, ty - 1] + img[tx + 1, ty - 1] + img[tx - 1, ty + 1] + img[tx - 1, ty - 1] + img[tx + 1, ty + 1]) / 9
        return img

    for p in range(args.divide):
        for q in range(args.divide):
            x = p * width
            y = q * width
            for k in range(width):
                img = smooth_pix(img, x + k, y + width - 1)
                if q != args.divide - 1:
                    img = smooth_pix(img, x + k, y + width)
            for k in range(width):
                img = smooth_pix(img, x + width - 1, y + k)
                if p != args.divide - 1:
                    img = smooth_pix(img, x + width, y + k)
    return img

def save_img(res, imgid, divide=False):
    output = res.detach().cpu().numpy() # d * d, 3, width, width    
    output = np.transpose(output, (0, 2, 3, 1))
    if divide:
        output = small2large(output)
        output = smooth(output)
    else:
        output = output[0]
    output = (output * 255).astype('uint8')
    output = cv2.resize(output, origin_shape)
    cv2.imwrite('output/generated' + str(imgid) + '.png', output)

actor = ResNet("actor", 3, 18, args.k * ACTION_SPACE) # action_bundle = 5, 65 = 5 * 13
actor.load_state_dict(torch.load(args.actor))
actor = actor.to(device).eval()
Painter = Painter.to(device).eval()

canvas = torch.zeros([1, 1, width, width]).to(device)

patch_img = cv2.resize(img, (width * args.divide, width * args.divide))
patch_img = large2small(patch_img)
patch_img = np.transpose(patch_img, (0, 3, 1, 2))
patch_img = torch.tensor(patch_img).to(device).float() / 255.

img = cv2.resize(img, (width, width))
img = img.reshape(1, width, width, 1)
img = np.transpose(img, (0, 3, 1, 2))
img = np.where(img > 100, 255, 0).astype(np.uint8)
img = torch.tensor(img).to(device).float() / 255.

os.system('mkdir output')

with torch.no_grad():
    if args.divide != 1:
        args.max_step = args.max_step // 2
    for i in range(args.max_step):
        stepnum = T * i / args.max_step
        actions = actor(torch.cat([canvas, img, stepnum], 1))
        canvas, res = Painter.paint(actions, canvas, args.k)
        print('canvas step {}, L2Loss = {}'.format(i, ((canvas - img) ** 2).mean()))
        for j in range(args.k):
            save_img(res[j], args.imgid)
            args.imgid += 1
    if args.divide != 1:
        canvas = canvas[0].detach().cpu().numpy()
        canvas = np.transpose(canvas, (1, 2, 0))    
        canvas = cv2.resize(canvas, (width * args.divide, width * args.divide))
        canvas = large2small(canvas)
        canvas = np.transpose(canvas, (0, 3, 1, 2))
        canvas = torch.tensor(canvas).to(device).float()
        T = T.expand(canvas_cnt, 1, width, width)
        for i in range(args.max_step):
            stepnum = T * i / args.max_step
            actions = actor(torch.cat([canvas, patch_img, stepnum], 1))
            canvas, res = Painter.paint(actions, canvas)
            print('divided canvas step {}, L2Loss = {}'.format(i, ((canvas - patch_img) ** 2).mean()))
            for j in range(5):
                if args.imgid < 50 or args.imgid % 50 == 0:
                    save_img(res[j], args.imgid, True)
                args.imgid += 1

os.system('rm videos/video.mp4')
os.system('ffmpeg -r 30 -f image2 -i output/generated%d.png -s 512x512 -c:v libx264 -pix_fmt yuv420p videos/video.mp4 -q:v 0 -q:a 0')