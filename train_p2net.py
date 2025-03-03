# basic lib
import os
import torch
import numpy as np
import os.path as osp

# data & options
from options.opt_p2net import Options
from torch.utils.data import DataLoader
from data.common import P2NETCollate
from data.dl_p2net import P2NETMatchDataset

# networks & loss func
from networks.d2f import D2F
from networks.d3f import KPFCNN
from networks.loss_p2net import p2net_criterion

# tools
import time
import datetime
from tensorboardX import SummaryWriter

# Config
opt = Options().parse()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
cuda = torch.cuda.is_available()
opt.device = "cuda:0" if cuda else "cpu"
writer = SummaryWriter(log_dir=opt.runs_dir)


def validation(d2net, d3net, val_loader, total_epoch, loss_mode):
    d2net.eval()
    d3net.eval()
    val_steps = 0
    tot_d_pos = 0
    tot_d_neg = 0
    tot_d_neg_row = 0
    tot_d_neg_col = 0
    tot_val_acc = 0
    tot_val_loss = 0
    toto_val_det_loss = 0
    print("****************************************************************")
    print("********************* Start validation {:d} ***********************".format(total_epoch + 1))
    print("****************************************************************\n")

    for val_data in val_loader:
        val_data.to(opt.device)
        with torch.set_grad_enabled(False):
            d2_out = d2net(val_data.images, val_data.valid_depth_mask, opt)
            d3_out = d3net(val_data, opt)
            val_loss, val_det_loss, val_acc, d_pos, d_neg, d_neg_row, d_neg_col = p2net_criterion(d2_out, d3_out, val_data.pos_keypts_inds, opt, 1, loss_mode)

            if val_acc > 0:
                val_steps += 1
                tot_d_pos += d_pos.item()
                tot_d_neg += d_neg.item()
                tot_d_neg_row += d_neg_row
                tot_d_neg_col += d_neg_col
                tot_val_acc += val_acc
                tot_val_loss += val_loss.item()
                toto_val_det_loss += val_det_loss.item()
            if val_steps == opt.val_size:
                break

    if val_steps > 0:
        mean_acc = tot_val_acc / val_steps
        mean_loss = tot_val_loss / val_steps
        mean_det_loss = toto_val_det_loss / val_steps
        mean_d_pos = tot_d_pos / val_steps
        mean_d_neg = tot_d_neg / val_steps
        mean_d_neg_row = tot_d_neg_row / val_steps
        mean_d_neg_col = tot_d_neg_col / val_steps
    else:
        mean_acc = 0
        mean_loss = 0
        mean_det_loss = 0
        mean_d_pos = 0
        mean_d_neg = 0
        mean_d_neg_row = 0
        mean_d_neg_col = 0

    print('------------------- Validation of epoch {:d} ----------------------'.format(total_epoch + 1))
    print("d_pos:{:f}, d_neg:{:f}, d_neg_row {:f}, d_neg_col {:f}".format(mean_d_pos, mean_d_neg, mean_d_neg_row, mean_d_neg_col))
    print("mean_loss:{:f}, mean_det_loss {:f}, mean_acc:{:f}".format(mean_loss, mean_det_loss, mean_acc))
    print("total validation steps:", val_steps)
    writer.add_scalar('val_acc', mean_acc, total_epoch)
    writer.add_scalar('val_loss', mean_loss, total_epoch)
    writer.add_scalar('val_det_loss', mean_det_loss, total_epoch)
    writer.add_scalar('val_d_pos', mean_d_pos, total_epoch)
    writer.add_scalar('val_d_neg', mean_d_neg, total_epoch)
    writer.add_scalar('val_d_neg_row', mean_d_neg_row, total_epoch)
    writer.add_scalar('val_d_neg_col', mean_d_neg_col, total_epoch)
    print("\n****************************************************************")
    print("********************* End of validation {:d} **********************".format(total_epoch + 1))
    print("****************************************************************\n")
    torch.cuda.empty_cache()
    return mean_acc


def train(d2net, d3net, train_loader, val_loader=None, ckpt=None):

    param_list = list(d2net.parameters()) + list(d3net.parameters())
    if opt.optm_type == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, param_list), lr=opt.sgd_lr,
                                    momentum=opt.sgd_momentum, weight_decay=opt.sgd_weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1 ** (2 / opt.epochs))
    elif opt.optm_type == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, param_list), lr=opt.adam_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1 ** (1 / 240000))
    else:
        raise NotImplementedError

    if opt.train_mode == 'continue':
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    best_acc = 0
    total_steps = 0
    epoch_acc = []
    epoch_loss = []
    epoch_det_loss = [] 
    epoch_d_pos = []
    epoch_d_neg = []
    epoch_d_neg_row = []
    epoch_d_neg_col = []
    loss_mode = 'Single'
    for epoch in range(opt.epochs):
        print("******************************* Start epoch {:d} ************************************".format(epoch + 1))
        d2net.train()
        d3net.train()
        if epoch == opt.switch_epoch:
            loss_mode = opt.loss_mode

        for train_data in train_loader:
            train_data.to(opt.device)
            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                d2_out = d2net(train_data.images, train_data.valid_depth_mask, opt)
                d3_out = d3net(train_data, opt)
                train_loss, train_det_loss, train_acc, d_pos, d_neg, d_neg_row, d_neg_col = p2net_criterion(d2_out, d3_out, train_data.pos_keypts_inds, opt,
                                                                                                          total_steps, loss_mode)

                # backward
                train_loss.backward()

                do_step = True
                for param in d3net.parameters():
                    if param.grad is not None:
                        if (1 - torch.isfinite(param.grad).long()).sum() > 0:
                            do_step = False
                            break

                if do_step is True:
                    if opt.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_value_(d3net.parameters(), opt.grad_clip_norm)
                    optimizer.step()
                    scheduler.step()

                epoch_acc.append(train_acc)
                epoch_loss.append(train_loss.item())
                epoch_det_loss.append(train_det_loss.item())
                epoch_d_pos.append(d_pos.item())
                epoch_d_neg.append(d_neg.item())
                epoch_d_neg_row.append(d_neg_row.item())
                epoch_d_neg_col.append(d_neg_col.item())

            total_steps += 1
            if total_steps == 1 or total_steps % opt.print_freq == 0:
                tmp_acc = np.mean(epoch_acc)
                tmp_loss = np.mean(epoch_loss)
                tmp_det_loss = np.mean(epoch_det_loss)
                tmp_d_pos = np.mean(epoch_d_pos)
                tmp_d_neg = np.mean(epoch_d_neg)
                tmp_d_neg_row = np.mean(epoch_d_neg_row)
                tmp_d_neg_col = np.mean(epoch_d_neg_col)
                epoch_acc = []
                epoch_loss = []
                epoch_det_loss = [] 
                epoch_d_pos = []
                epoch_d_neg = []
                epoch_d_neg_row = []
                epoch_d_neg_col = []
                writer.add_scalar('train_acc', tmp_acc, total_steps)
                writer.add_scalar('train_loss', tmp_loss, total_steps)
                writer.add_scalar('train_det_loss', tmp_det_loss, total_steps)
                writer.add_scalar('train_d_pos', tmp_d_pos, total_steps)
                writer.add_scalar('train_d_neg', tmp_d_neg, total_steps)
                writer.add_scalar('train_d_neg_row', tmp_d_neg_row, total_steps)
                writer.add_scalar('train_d_neg_col', tmp_d_neg_col, total_steps)
                print('Train: total_steps {:d}, train_loss {:f}, train_det_loss {:f} train_acc {:f}, d_pos {:f}, d_neg {:f}, d_neg_row {:f}, d_neg_col {:f}'.format(total_steps,
                                                                                                                                                                    tmp_loss,
                                                                                                                                                                    tmp_det_loss,
                                                                                                                                                                    tmp_acc,
                                                                                                                                                                    tmp_d_pos,
                                                                                                                                                                    tmp_d_neg,
                                                                                                                                                                    tmp_d_neg_row,
                                                                                                                                                                    tmp_d_neg_col))

        # process validation after each epoch
        tmp_val_acc = validation(d2net, d3net, val_loader, epoch, loss_mode)
        d2net.train()
        d3net.train()
        if epoch > 4 and epoch % opt.save_freq == 0:
            print(f"save model of epoch {epoch + 1}")
            filename = osp.join(opt.models_dir, f'ckpt_{epoch + 1}.pth.tar')
            checkpoint_dict = {'epoch': epoch + 1,
                            'd2net_state_dict': d2net.state_dict(),
                            'd3net_state_dict': d3net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}
            torch.save(checkpoint_dict, filename)

            # save best model
            if tmp_val_acc > best_acc:
                print(f"save best model")
                best_filename = osp.join(opt.models_dir, f'best.pth.tar')
                best_acc = tmp_val_acc
                torch.save(checkpoint_dict, best_filename)

        # reconstruct the batched data index after each epoch
        val_loader.dataset.reset_batch_list(opt.batchsize)
        train_loader.dataset.reset_batch_list(opt.batchsize)


def main():
    # Load the dataset
    t1 = time.time()
    print("******* Prepare dataset *******")
    test_set = P2NETMatchDataset(opt, 'test')
    train_set = P2NETMatchDataset(opt, 'train')

    kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}
    test_loader = DataLoader(test_set, batch_size=1, collate_fn=P2NETCollate, shuffle=True, **kwargs)
    train_loader = DataLoader(train_set, batch_size=1, collate_fn=P2NETCollate, shuffle=True, **kwargs)

    t2 = time.time()
    print("\nDone in {:f} seconds".format(t2 - t1))

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    # Models
    print("******* Creating model *******")
    d2net = D2F(opt)
    d3net = KPFCNN(opt)
    d2net.to(opt.device)
    d3net.to(opt.device)

    if opt.train_mode == 'from-scratch':
        print("Training mode: from scratch...")

    elif opt.train_mode == 'fine-tune':
        print("Training mode: with pre-trained models...")
        d2net_weights = f'./logs/{opt.d2net_logs}/models/{opt.d2net_ckpt}'
        d3net_weights = f'../CDF-master/logs/{opt.d3net_logs}/models/{opt.d3net_ckpt}'
        d2net_ckpt = torch.load(d2net_weights, map_location=opt.device)
        d3net_ckpt = torch.load(d3net_weights, map_location=opt.device)

        d2net.load_state_dict(d2net_ckpt['d2net_state_dict'])
        d3net.load_state_dict(d3net_ckpt['d3net_state_dict'])

    elif opt.train_mode == 'continue':
        print("Training mode: with pre-trained p2p weights...")
        weights = f'./logs/{opt.p2p_logs}/models/{opt.p2p_ckpt}'
        ckpt = torch.load(weights, map_location=opt.device)
        print(f"Loaded models from {opt.p2p_ckpt}")

        d2net.load_state_dict(ckpt['d2net_state_dict'])
        d3net.load_state_dict(ckpt['d3net_state_dict'])

    else:
        raise ValueError

    print(d2net)
    print(d3net)
    
    print(get_parameter_number(d2net))
    print(get_parameter_number(d3net))

    t3 = time.time()
    print("\nDone in {:f} seconds".format(t3 - t2))

    # Train
    print("\n******* Start training *******")
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("******* Start time: {:s} ********".format(start_time))

    if opt.train_mode == 'from-scratch':
        train(d2net, d3net, train_loader, test_loader)
    elif opt.train_mode == 'continue':
        # train(d2net, d3net, train_loader, val_loader, ckpt)
        validation(d2net, d3net, test_loader, 0)
    else:
        raise NotImplementedError

    # validation(d2net, d3net, val_loader, 0, opt.loss_mode)
    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n******* Finished at {:s} *******".format(end_time))
    writer.close()


if __name__ == '__main__':
    main()
