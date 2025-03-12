from model.coattnet import coattnet_v2_withWeighted_tiny
from utils.dataset import loader
import torch
from config import conf
from torch.optim import lr_scheduler
import torch.nn as nn
import os
from tools.train_model import train_model
from tools.focal_loss import FocalLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = coattnet_v2_withWeighted_tiny(num_classes=conf.task_layer.num_classes).to(device)
train_loader = loader(train=True)   # 训练集
eval_loader = loader(train=False)    # 验证集
criterion = FocalLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=conf.train.lr, weight_decay=0.005)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=conf.train.epochs, eta_min=0)
run_id = "{}_{}_{}-{}-focal_loss".format(
        conf.image_size[0], conf.train.epochs, conf.batch_size,
        conf.train.lr
    )
if conf.train.resume:
        if os.path.isfile(conf.train.resume):
            print("=> resume is True. ====\n====loading checkpoint '{}'".format(conf.train.resume))
            checkpoint = torch.load(conf.train.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            print('Model loaded from {}'.format(conf.train.resume))
        else:
            raise ValueError(" ??? no checkpoint found at '{}'".format(conf.train.resume))
else:
    start_epoch = 0
    start_step = 0
train_model(model, train_loader, eval_loader,
                criterion, optimizer, scheduler,
                conf.batch_size, num_epochs=conf.train.epochs,
                start_epoch=start_epoch, start_step=start_step,
                task="multi_labels", eval_interval=conf.train.eval_interval,
                run_id=run_id,
                device=device,
                test_loader=None)