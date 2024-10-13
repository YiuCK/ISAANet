import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
import math
import numpy as np
import os, argparse
from datetime import datetime
from data import get_loader
from func import AvgMeter
from torch import nn
from network import Net
import timm

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=8e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=6, help='batch size')
parser.add_argument('--trainsize', type=int, default=352, help='input size')
parser.add_argument('--lanmb', type=float, default=3, help='lanmb1')
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
image_root = './ORSSD/train/img/'
gt_root = './ORSSD/train/gt/'
save_path = './model/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5*(1+math.cos(t*math.pi/T))*init_lr
    return lr

def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)


BCE = nn.BCEWithLogitsLoss().cuda()
DL = IOU(size_average=True).cuda()

pretrained_cfg = timm.models.create_model('mobilenetv3_large_100').default_cfg
pretrained_cfg['file'] = r'./mobilenetv3_large_100_ra-f55367f5.pth'

model = Net(pretrained_cfg)
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params=filter(lambda x: x.requires_grad, params), lr=opt.lr, betas=(0.5, 0.999), eps=1e-08)

size_rates = [0.75, 1, 1.25]  # multi-scale training
T = opt.epoch

if __name__ == '__main__':
    for epoch in range(1, opt.epoch + 1):
        model.train()
        loss0_record, loss_record1, loss_record2, loss_record3, loss_record4, loss_record5, loss_record6, loss_record7, \
            loss_record8, loss_record9, loss_record10, loss_record11, loss_record12 = \
            AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), \
                AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        lr = lr_schedule_cosdecay(epoch, T)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()

                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()

                trainsize = int(round(opt.trainsize * rate / 32) * 32)
                if rate != 1:
                    images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                s1, s2, s3 = model(images)
                loss1 = BCE(s1, gts) + opt.lanmb * DL(torch.sigmoid(s1), gts)
                loss2 = BCE(s2, gts) + opt.lanmb * DL(torch.sigmoid(s2), gts)
                loss3 = BCE(s3, gts) + opt.lanmb * DL(torch.sigmoid(s3), gts)

                loss = loss1 + loss2 + loss3

                loss.backward()

                optimizer.step()
                if rate == 1:
                    loss0_record.update(loss.data, opt.batchsize)
                    loss_record1.update(loss1.data, opt.batchsize)
                    loss_record2.update(loss2.data, opt.batchsize)
                    loss_record3.update(loss3.data, opt.batchsize)

            if i % 50 == 0 or i == total_step:
                log = '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LossA: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}'. \
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss0_record.avg, loss_record1.avg,
                           loss_record2.avg, loss_record3.avg)
                print(log)


        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_path + '%d' % epoch + '.pth')
    torch.save(model.state_dict(), save_path + 'final.pth')


