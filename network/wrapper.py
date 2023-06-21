import os
from argparse import Namespace
import numpy as np
import torch
from utils.metrics import eval_metrics
from utils.setup_helper import load_weights
from network import losses
from data.gden import *

class NetworkWrapper:
    def __init__(self, net, iter_per_epoch, opt, config):
        self.net = net
        self.iter_per_epoch = iter_per_epoch
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device(f'cuda:{self.gpu_ids[0]}') if self.gpu_ids else torch.device('cpu')
        optim_dict, start_epoch = self.load_ckpt(opt)
        opt.start_epoch = start_epoch
        self.set_optimizer(opt, config, optim_dict)
        self.best_acc = 0.0
        self.config = config
        self.semantic_seg_loss = losses.create_loss(self.config['LOSS'])
        self.num_classes = self.config['NETWORK']['OUTPUT_CHANNELS']

    def set_optimizer(self, opt, config, optim_dict):
        if not opt.is_Train:
            return

        # Initialize optimizer
        lr = config['TRAINING']['LR']
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        print(f'Setup Adam optimizer(lr={lr})')

        # Reload optimizer state dict if exists
        if optim_dict:
            self.optimizer.load_state_dict(optim_dict)

    def recursive_todevice(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            return [self.recursive_todevice(c) for c in x]
        elif isinstance(x, dict):
            return {k: self.recursive_todevice(v) for k, v in x.items()}
        else:
            return x.to(self.device)

    def optim_step_(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_epoch(self, epoch, data_loader, log_print):
        self.net.train()
        epoch_loss = []
        for i, data in enumerate(data_loader):
            data = self.recursive_todevice(data)
            y = data["region"].y.argmax(-1).long() # Take the majority class of the region as label
            pred = self.net(data)
            loss = self.semantic_seg_loss(pred, y)
            self.optim_step_(loss)
            epoch_loss.append(loss.item())
            if (i+1) % 50 == 0 or (i+1) == len(data_loader):
                log_print(f'Batch:{i + 1} TRAIN loss={np.mean(epoch_loss):.3f}')

    def eval_model(self, epoch, data_loader, log_print):
        """Evaluate the model
                METRICS: nodeAcc, meanIoU, classIoU, loss
        """
        self.net.eval()
        epoch_val_loss = []
        total_correct, total_label = 0, 0
        total_inter, total_union = 0, 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                data = self.recursive_todevice(data)
                y = data["region"].y.argmax(-1).long() # Take the majority class of the region as label
                pred = self.net(data)
                val_loss = self.semantic_seg_loss(pred, y)
                epoch_val_loss.append(val_loss.item())
                correct, labeled, inter, union = eval_metrics(pred, y, self.num_classes, self.config['LOSS']['IGNORE_INDEX'])
                total_inter, total_union = total_inter + inter, total_union + union
                total_correct, total_label = total_correct + correct, total_label + labeled

                # PRINT INFO
                nodeAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()

                seg_metrics = {"Node_Accuracy": np.round(nodeAcc, 3), "Mean_IoU": np.round(mIoU, 3),
                           "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))}

            metrics = Namespace(nodeAcc=nodeAcc, mIoU=mIoU, c=seg_metrics["Class_IoU"], loss=np.mean(epoch_val_loss))
        return metrics

    def save_ckpt(self, epoch, out_dir, last_ckpt=False, best_acc=None, is_best=False):
        ckpt = {'last_epoch': epoch, 'best_acc': best_acc, 'model_dict': self.net.state_dict(),
                'optimizer_dict': self.optimizer.state_dict()} # NOTE: should save the state of a LR scheduler if you want to use one

        if last_ckpt:
            ckpt_name = 'last_ckpt.pth'
        elif is_best:
            ckpt_name = 'best_ckpt.pth'
        else:
            ckpt_name = f'ckpt_ep{epoch + 1}.pth'
        ckpt_path = os.path.join(out_dir, ckpt_name)
        torch.save(ckpt, ckpt_path)

    def load_ckpt(self, config):
        ckpt_path = config.checkpoint
        if ckpt_path is None:
            return None, 0

        ckpt = load_weights(ckpt_path, self.device)
        start_epoch = ckpt['last_epoch'] + 1
        self.best_acc = ckpt['best_acc']
        print(f'Load ckpt from {ckpt_path}, reset start epoch {config.start_epoch}, best acc {self.best_acc}')

        # Load net state
        model_dict = ckpt['model_dict']
        if len(model_dict.items()) == len(self.net.state_dict()):
            print('Reload all net parameters from weights dict')
            self.net.load_state_dict(model_dict)
        else:
            print('Reload part of net parameters from weights dict')
            self.net.load_state_dict(model_dict, strict=False)

        # NOTE: should load the state of a LR scheduler if you want to use one

        # Load optimizer state
        return ckpt['optimizer_dict'], start_epoch
