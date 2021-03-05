import torch, sys, torchvision
import torch.nn as nn
import skimage.metrics as metrics
import numpy as np
import densetorch as dt
import pdb
import torch.nn.functional as F

from torch.utils.data import DataLoader
from unet import UNet
from pytorch_lightning.core.lightning import LightningModule
from dataio import transform_train, transform_val
from config import *


class InvHuberLoss(nn.Module):
    """Inverse Huber Loss for depth estimation.
    The setup is taken from https://arxiv.org/abs/1606.00373
    Args:
      ignore_index (float): value to ignore in the target
                            when computing the loss.
    """

    def __init__(self, ignore_index=0):
        super(InvHuberLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, x, target):
        input = F.relu(x)  # depth predictions must be >=0
        diff = input - target
        mask = target != self.ignore_index

        err = torch.abs(diff * mask.float())
        c = 0.2 * torch.max(err)
        err2 = (diff ** 2 + c ** 2) / (2.0 * c)
        mask_err = err <= c
        mask_err2 = err > c
        cost = torch.mean(err * mask_err.float() + err2 * mask_err2.float())
        return cost



class ImageUnet(nn.Module):
    def __init__(self):
        super().__init__()

        base_ch = 3
        n_layers = 4
        output_layer = nn.Sequential(
            nn.Conv2d(base_ch, 3, kernel_size=1, bias=True)
        )

        self.net = nn.Sequential(
            UNet(channels=[base_ch, base_ch, 2 * base_ch, 2 * base_ch, 4 * base_ch, 4 * base_ch],
                 n_layers=n_layers))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        result = self.net(img)
        return result

def to3(x):
    return torch.cat((x,x,x), dim=0).unsqueeze(0)

class Model(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.train_dataset = dt.data.MMDataset(data_file, data_dir, line_to_paths_fn, masks_names, transform=transform_train)
        self.val_dataset = dt.data.MMDataset(val_file, data_val_dir, line_to_paths_fn, masks_names, transform=transform_val)
        self.encoder = ImageUnet()
        self.seg_dec3 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, padding=1)

        self.depth_dec3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1)
        self.lr = args.lr
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.depth_loss = InvHuberLoss(ignore_index=ignore_depth)
        self.batch_size = 1
        self.steps = 0



    def get_seg_loss(self, result, gt):
        return self.seg_loss(result, gt)

    def get_depth_loss(self, result, gt):
        return self.depth_loss(result, gt)

    def training_step(self, batch, batch_idx):
        x = batch['image'].float() # [1, 3, 400, 400], float64
        seg = batch['segm'] # [1, 400, 400], int64
        depth = batch['depth'] # [1, 400, 400], float32
        seg_pred, dep_pred = self.encoder(x)
        seg_pred = self.seg_dec3(seg_pred)
        depth_pred = self.depth_dec3(dep_pred)

        seg_loss = self.get_seg_loss(seg_pred, seg)
        depth_loss = self.get_depth_loss(depth_pred, depth)


        total_loss = depth_loss + seg_loss

        x = x.float()

        seg = seg.float()


        seg_pred = torch.argmax(seg_pred, dim=1).float()
        depth_pred = depth_pred.squeeze(0)


        seg_pred = to3(seg_pred)
        depth_pred = to3(depth_pred)
        seg = to3(seg)
        depth = to3(depth)

        gt = torch.cat((x, seg, depth), dim=0)
        grid = torchvision.utils.make_grid(gt,
                                           scale_each=True,
                                           nrow=self.batch_size,
                                           normalize=False).cpu().detach().numpy()
        self.logger.experiment.add_image("gt", grid, self.steps)


        pred = torch.cat((x, seg_pred, depth_pred), dim=0)
        grid = torchvision.utils.make_grid(pred,
                                           scale_each=True,
                                           nrow=self.batch_size,
                                           normalize=False).cpu().detach().numpy()
        self.logger.experiment.add_image("pred", grid, self.steps)
        self.steps += 1

        self.log('train_loss', total_loss,
                 on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=val_batch_size,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True,
                          drop_last=False)

