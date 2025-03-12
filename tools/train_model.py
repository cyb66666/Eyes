#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import torch
import time
from tensorboardX import SummaryWriter
from tools.eval_model import eval_model
from tools.eval import eval
from tqdm import tqdm


def train_model(model, train_loader, eval_loader,
                criterion, optimizer, scheduler,
                batch_size, num_epochs=5,
                start_epoch=0, start_step=0,
                task="multi_classes",
                eval_interval=5,
                run_id="run_id",
                device=torch.device("cuda:0"),
                test_loader=None):
    model.to(device=device)
    criterion.to(device=device)
    tot_step_count = start_step

    best_acc = -1.

    dir_checkpoint = "./results/" + run_id
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    # logdir = "./runs/" + run_id
    # writer = SummaryWriter(log_dir=logdir)

    log_txt_path = "./results/" + run_id + "/evalLog.csv"
    with open(log_txt_path, 'a') as f:
        f.write("epoch, acc, precision_cl, recall_c, precision, recall, loss, {}\n".format(time.asctime(time.localtime(time.time()))))

    for epoch in range(start_epoch, num_epochs):
        localtime = time.asctime(time.localtime(time.time()))
        print('{} - Starting epoch {}/{}.\n'.format(localtime, epoch, start_epoch + num_epochs))

        model.train()
        for train_item in tqdm(train_loader):
            def closure():
                optimizer.zero_grad()
                inputs, label = train_item
                inputs = inputs.to(device=device, dtype=torch.float)
                # label = torch.tensor(np.array(label, dtype=float)).to(device=device)
                label = label.to(device=device)
                outputs = model(inputs)
                # 2 outputs if distillation token is adopted
                if isinstance(outputs, tuple):
                    loss = criterion(outputs[0], label)
                else:
                    loss = criterion(outputs, label)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            # writer.add_scalar('loss', loss.item(), global_step=tot_step_count)
            # writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=tot_step_count)
            tot_step_count += 1
        # updating lr
        scheduler.step()

        if (epoch) % eval_interval == 0:
            eval_re = eval(model, eval_loader, criterion)
            # writer.add_scalar('eval-loss', eval_re["loss"], global_step=epoch)
            # writer.add_scalar('eval-acc', eval_re["accuracy"], global_step=epoch)
            # writer.add_scalar('eval-precision-c', eval_re["precision_per_class"], global_step=epoch)
            # writer.add_scalar('eval-recall-c', eval_re["recall_per_class"], global_step=epoch)
            # writer.add_scalar('eval-recall', eval_re["macro_recall"], global_step=epoch)
            # writer.add_scalar('eval-recall', eval_re["macro_recall"], global_step=epoch)

            with open(log_txt_path, 'a') as f:
                f.write(str(epoch) + "," + ",".join([str(eval_re_i) for eval_re_i in list(eval_re.values())]) + ",")
            print("finished eval. epoch={} \n= = = \n".format(epoch))
            print("saveing checkpoint ...")
            with open(log_txt_path, 'a') as f:
                f.write("save state!")

            if eval_re["accuracy"] > best_acc:
                print("best acc is {}".format(eval_re["accuracy"]))
                with open(log_txt_path, 'a') as f:
                    f.write("is best acc and save state! \n")
                best_acc = eval_re["accuracy"]
                state = {'epoch': epoch,
                         'step': tot_step_count,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                torch.save(state, os.path.join(dir_checkpoint, run_id + '.pth.tar'))
            else:
                with open(log_txt_path, 'a') as f:
                    f.write("\n")
    # writer.close()

    if test_loader is not None:
        # load best model
        load_from_path = os.path.join(dir_checkpoint, run_id + '.pth.tar')
        print("=> TESTing... ====\n====loading checkpoint '{}'".format(load_from_path))
        checkpoint = torch.load(load_from_path)
        save_epoch = checkpoint['epoch']
        # save_step = checkpoint['step']
        miss, unexp = model.load_state_dict(checkpoint['state_dict'])
        print('Model loaded from {}, \nmiss={}\nunexp={}'.format(load_from_path, miss, unexp))
        test_re = eval_model(model, test_loader, criterion=criterion, device=device,
                             task=task)
        # log
        with open(log_txt_path, 'a') as f:
            f.write("test-" + str(save_epoch) + "," + ",".join(
                [str(test_re_i) for test_re_i in list(test_re.values())]) + ",")
        # print("eval-{}: {}".format(save_epoch, "".join(["{}:{:.4f}, ".format(k, v) for k,v in eval_re.items() ])))
        print("finished test. epoch={} \n= = = \n".format(save_epoch))
