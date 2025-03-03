import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.libs.d2_lib import D2, Ops


class D2E(nn.Module):
    def __init__(self, in_dim):
        super(D2E, self).__init__()
        # conv = conv2d + BN + ReLU
        self.conv1 = D2.conv(in_dim, 32, 3, 1)
        self.conv2 = D2.conv(32, 32, 3, 1)

        self.conv3 = D2.conv(32, 64, 3, 1)
        self.conv4 = D2.conv(64, 64, 3, 1, dilation=2, padding=2)

        self.conv5 = D2.conv(64, 128, 3, 1, dilation=2, padding=2)
        self.conv6 = D2.conv(128, 128, 3, 1, dilation=4, padding=4)

        self.conv7_1 = D2.conv(128, 128, 3, 1, dilation=4, padding=4, relu=False)
        self.conv7_2 = D2.conv(128, 128, 3, 1, dilation=8, padding=8, relu=False)
        self.conv7_3 = D2.conv(128, 128, 3, 1, dilation=16, padding=16, relu=False)

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)

        output = self.conv3(output)
        output = self.conv4(output)

        output = self.conv5(output)
        output = self.conv6(output)

        output = self.conv7_1(output)
        output = self.conv7_2(output)
        feature_map = self.conv7_3(output)

        return feature_map


class D2F(nn.Module):
    def __init__(self, opt):
        super(D2F, self).__init__()
        self.opt = opt
        if self.opt.img_mode == 'color':
            in_dim = 3
        elif self.opt.img_mode == 'gray':
            in_dim = 1
        else:
            print("Wrong image type!")
            raise ValueError

        self.feature_extractor = D2E(in_dim)

    def forward(self, inputs, valid_depth_mask, opt):
        feature_map = self.feature_extractor(inputs)
        alpha, beta = Ops.peakiness_score(feature_map, 1, opt.isTrain)
        score_map = torch.max(alpha * beta, dim=1, keepdim=True)[0]

        if opt.isTrain:
            return feature_map, score_map
        else:
            descs_batch = {'pred':[], 'rand':[]}
            kpts_batch = {'pred':[], 'rand':[]}
            scores_batch = {'pred':[], 'rand':[]}
            for idy in range(opt.batchsize):
                kpt_inds, kpt_score = Ops.pixel_hard_selection(
                    score_map[idy:idy+1], feature_map[idy:idy+1], valid_depth_mask=valid_depth_mask[idy][:, :, 0], k=opt.kpt_n,
                    score_thld=opt.score_thld, edge_thld=opt.edge_thld,
                    nms_size=opt.nms_size, eof_size=opt.eof_mask, device=opt.device)

                kpt_inds = kpt_inds.to(torch.float32)

                # get pred pixels
                # print("d2 feature map before norm:", feature_map)
                descs = F.normalize(feature_map[idy][:, kpt_inds[0][0, :].long(), kpt_inds[0][1, :].long()], dim=0)
                kpts = torch.stack([kpt_inds[:, 1, :], kpt_inds[:, 0, :]], dim=-1).squeeze(0)
                scores = kpt_score.squeeze(0)

                descs_batch['pred'].append(descs.transpose(1, 0).cpu().numpy())
                kpts_batch['pred'].append(kpts.cpu().numpy())
                scores_batch['pred'].append(scores.cpu().numpy())

                # get rand pixels
                valid_coords = torch.nonzero(valid_depth_mask[idy])
                random_idx = torch.randperm(len(valid_coords))[:opt.kpt_n]
                random_coords = valid_coords[random_idx].transpose(1, 0).long()
                # random_h = torch.randint(0, 479, (1, opt.kpt_n))
                # random_w = torch.randint(0, 639, (1, opt.kpt_n))
                # random_coords = torch.cat((random_h, random_w), dim=0)

                rand_descs = F.normalize(feature_map[idy][:, random_coords[0, :], random_coords[1, :]], dim=0)
                rand_kpts = torch.stack([random_coords[1, :], random_coords[0, :]], dim=-1)
                rand_scores = score_map[idy][:, random_coords[0, :], random_coords[1, :]]

                descs_batch['rand'].append(rand_descs.transpose(1, 0).cpu().numpy())
                kpts_batch['rand'].append(rand_kpts.cpu().numpy())
                scores_batch['rand'].append(rand_scores.cpu().numpy())

            return descs_batch, kpts_batch, scores_batch

    def finetune(self, finetune_layers):
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        for layer_name in finetune_layers:
            for name, child in self.feature_extractor.named_children():
                if layer_name == name:
                    for param in child.parameters():
                        param.requires_grad = True
        print(f"D2 finetune layers : {finetune_layers}")
