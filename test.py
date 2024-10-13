import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np
import os
from scipy import misc
from data import test_dataset
from network import Net
from PIL import Image
import timm
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

pretrained_cfg = timm.models.create_model('mobilenetv3_large_100').default_cfg
pretrained_cfg['file'] = r'./mobilenetv3_large_100_ra-f55367f5.pth'


model = Net(pretrained_cfg)

model.load_state_dict(torch.load('./ORSSD.pth'))
model.cuda()
model.eval()

data_path = './'
valset = ['ORSSD']

for dataset in valset:
    save_path = './sal/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = data_path + dataset + '/test/img/'
    gt_root = data_path + dataset + '/test/gt/'
    print(gt_root)
    test_loader = test_dataset(image_root, gt_root, testsize=352)

    with torch.no_grad():
        for i in range(test_loader.size):
            print(i)
            image, gt, name = test_loader.load_data()
            gt = np.array(gt).astype('float')
            gt = gt / (gt.max() + 1e-8)
            image = Variable(image).cuda()

            s1,s2,pre = model(image)
            res = F.interpolate(pre, size=gt.shape, mode='bilinear', align_corners=True)
            res = res.data.sigmoid().cpu().numpy().squeeze()
            res = Image.fromarray(np.uint8(255 * res)).convert('RGB')
            res.save(save_path + name + '.png')
