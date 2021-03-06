import numpy as np
import torch, sys, os
import torch.nn as nn
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from model import Model
import densetorch as dt
from config import *
from argparse import ArgumentParser

# model_options
print(torch.cuda.is_available())

def train(args):
    torch.cuda.empty_cache()

    model = Model(args)

    os.makedirs(f'./logs/{args.exp_name}', exist_ok=True)
    logger = TensorBoardLogger(save_dir='./logs/',
                               name=args.exp_name)

    checkpoint = pl.callbacks.ModelCheckpoint(monitor='train_loss_step',
                                              filepath=os.path.join(logger.log_dir, 'checkpoints',
                                                                    '{epoch:02d}-{train_loss:.3f}'),
                                              verbose=True,
                                              save_top_k=10,
                                              save_last=True,
                                              mode='min')

    trainer = pl.Trainer(fast_dev_run=False,
                         logger=logger,
                         max_epochs=args.max_epoch,
                         callbacks=[checkpoint],
                         gpus=0)
    trainer.fit(model)
    return

if __name__ == '__main__':
    parser = ArgumentParser()

    # data paths
    parser.add_argument('--local', default=False)
    parser.add_argument('--data_root', type=str,
                        help='path to folder with training data',
                        default='../data/MSR-Demosaicing/Dataset_LINEAR_without_noise/bayer_panasonic')
    parser.add_argument('--logging_root', type=str, default='.', required=False,
                        help='path to folder to store checkpoints and summaries')
    parser.add_argument('--download_data', type=bool, default=False, help='download data')
    parser.add_argument('--img_sidelength', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=1)

    # train params
    parser.add_argument('--train_test', type=str, default='train', help='Train or test evaluation')
    parser.add_argument('--exp_name', type=str, help='name of experiment', required=True)
    parser.add_argument('--max_epoch', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.001')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size to use')
    parser.add_argument('--reg_weight', type=int, default=0.05, help='loss regularization')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to restore a previous checkpoint')

    parser.add_argument('--fourier', type=bool, required=True, help='path to restore a previous checkpoint')

    args = parser.parse_args()
    #print('\n'.join(["%s: %s" % (key, value) for key, value in vars(args).items()]))

    train(args)

