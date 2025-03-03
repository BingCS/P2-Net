import torch
import torch.nn as nn
import torch.nn.functional as F


class D2(nn.Module):

    @staticmethod
    def conv(in_channels, out_channels, kernel_size, stride, padding=1, dilation=1, relu=True):

        if relu:
            conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                          padding, dilation, bias=False),
                # nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01),
                nn.LeakyReLU(0.1)
                # nn.ReLU()
            )
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, bias=True)

        return conv


class Ops(nn.Module):

    @staticmethod
    def peakiness_score(inputs, dilation=1, isTrain=True):
        ksize = dilation * 2 + 1

        max_per_sample = torch.max(inputs)
        if isTrain:
            inputs = inputs / max_per_sample
        else:
            inputs = inputs
        avg_inputs = F.avg_pool2d(F.pad(inputs, [dilation]*4, mode='reflect'), ksize, stride=1)

        alpha = F.softplus(inputs - avg_inputs)
        beta = F.softplus(inputs - torch.mean(inputs, dim=1, keepdim=True))

        return alpha, beta

    @staticmethod
    def d2net_score(inputs, dilation=1, training=True):
        b = inputs.size(0)
        ksize = dilation * 2 + 1

        max_per_sample = torch.max(inputs.view(b, -1), dim=1)[0]
        if training:
            exp = torch.exp(inputs / max_per_sample.view(b, 1, 1, 1))
        else:
            exp = torch.exp(inputs)
        sum_exp = (9 * F.avg_pool2d(F.pad(exp, [dilation]*4, mode='constant', value=1.), ksize, stride=1))
        depth_wise_max = torch.max(inputs, dim=1)[0]

        alpha = exp / sum_exp
        beta = inputs / depth_wise_max.unsqueeze(1)

        return alpha, beta

    @staticmethod
    def extract_kpts(score_map, valid_depth_mask, k, score_thld, edge_thld, nms_size, eof_size, device):
        b, c, h, w = score_map.size()

        mask = score_map > score_thld
        if nms_size > 0:
            nms_mask = F.max_pool2d(score_map, kernel_size=nms_size, stride=1, padding=1)
            nms_mask = torch.eq(score_map, nms_mask)
            mask = nms_mask & mask
        if eof_size > 0:
            eof_mask = torch.ones((1, 1, h - 2 * eof_size, w - 2 * eof_size), device=device, dtype=torch.float32)
            eof_mask = F.pad(eof_mask, [eof_size] * 4)
            eof_mask = eof_mask.bool()
            mask = eof_mask & mask
        if edge_thld > 0:
            edge_mask = Ops.edge_mask_deptehwise(score_map, 1, edge_thld, device)
            mask = edge_mask & mask

        mask = torch.reshape(mask, (h, w))
        score_map = torch.reshape(score_map, (h, w))
        mask = valid_depth_mask & mask

        indices = torch.nonzero(mask) ## check
        scores = score_map[indices[:, 0], indices[:, 1]]
        sample = torch.argsort(scores, descending=True)[0:k]

        indices = torch.unsqueeze(indices[sample].transpose(1, 0), dim=0)
        scores = torch.unsqueeze(scores[sample], dim=0)

        return indices, scores

    @staticmethod
    def pixel_hard_selection(score_map, feature_map, valid_depth_mask, k, score_thld, edge_thld, nms_size, eof_size, device):
        b, c, h, w = feature_map.size()

        # mask = score_map > score_thld
        mask = torch.ones_like(feature_map).bool()
        if nms_size > 0:
            local_max = F.max_pool2d(feature_map, kernel_size=nms_size, stride=1, padding=1)
            is_local_max = torch.eq(feature_map, local_max)
            depth_wise_max = torch.max(feature_map, dim=1, keepdim=True)[0]
            is_depth_wise_max = torch.eq(feature_map, depth_wise_max)
            nms_mask = is_local_max & is_depth_wise_max
            mask = nms_mask & mask
        if eof_size > 0:
            eof_mask = torch.ones((1, 1, h - 2 * eof_size, w - 2 * eof_size), device=device, dtype=torch.float32)
            eof_mask = F.pad(eof_mask, [eof_size] * 4)
            eof_mask = eof_mask.bool()
            mask = eof_mask & mask
        if edge_thld > 0:
            edge_mask = Ops.edge_mask_deptehwise(feature_map, 1, edge_thld, device)
            mask = edge_mask & mask

        # depth check
        mask = torch.max(mask, dim=1)[0]
        mask = torch.reshape(mask, (h, w))
        mask = valid_depth_mask & mask
        score_map = torch.reshape(score_map, (h, w))

        indices = torch.nonzero(mask) ## check
        print("d2 keypts:", len(indices), len(torch.nonzero(valid_depth_mask)))
        scores = score_map[indices[:, 0], indices[:, 1]]
        sample = torch.argsort(scores, descending=True)[0:k]

        indices = torch.unsqueeze(indices[sample].transpose(1, 0), dim=0)
        scores = torch.unsqueeze(scores[sample], dim=0)

        return indices, scores

    @staticmethod
    def kpt_refinement(inputs, device):
        b, c, h, w = inputs.size()
        di_filter = torch.tensor([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], device=device).view(1, 1, 3, 3)
        dj_filter = torch.tensor([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], device=device).view(1, 1, 3, 3)

        dii_filter = torch.tensor([[0, 1., 0], [0, -2., 0], [0, 1., 0]], device=device).view(1, 1, 3, 3)
        dij_filter = 0.25 * torch.tensor([[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]], device=device).view(1, 1, 3, 3)
        djj_filter = torch.tensor([[0, 0, 0], [1., -2., 1.], [0, 0, 0]], device=device).view(1, 1, 3, 3)
        dii = F.conv2d(inputs.view(-1, 1, h, w), dii_filter, padding=1).view(b, c, h, w)
        dij = F.conv2d(inputs.view(-1, 1, h, w), dij_filter, padding=1).view(b, c, h, w)
        djj = F.conv2d(inputs.view(-1, 1, h, w), djj_filter, padding=1).view(b, c, h, w)
        det = dii * djj - dij * dij

        inv_hess_00 = djj / det
        inv_hess_01 = -dij / det
        inv_hess_11 = dii / det

        di = F.conv2d(inputs.view(-1, 1, h, w), di_filter, padding=1).view(b, c, h, w)
        dj = F.conv2d(inputs.view(-1, 1, h, w), dj_filter, padding=1).view(b, c, h, w)

        step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
        step_j = -(inv_hess_01 * di + inv_hess_11 * dj)

        return torch.stack([step_i, step_j], dim=1)

    @staticmethod
    def edge_mask_deptehwise(inputs, dilation, edge_thld, device):
        b, c, h, w = inputs.size()
        dii_filter = torch.tensor([[0, 1., 0], [0, -2., 0], [0, 1., 0]], device=device).view(1, 1, 3, 3)
        dij_filter = 0.25 * torch.tensor([[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]], device=device).view(1, 1, 3, 3)
        djj_filter = torch.tensor([[0, 0, 0], [1., -2., 1.], [0, 0, 0]], device=device).view(1, 1, 3, 3)

        ## depth_wise_conv2d
        dii = F.conv2d(inputs.view(-1, 1, h, w), dii_filter, padding=dilation, dilation=dilation).view(b, c, h, w)
        dij = F.conv2d(inputs.view(-1, 1, h, w), dij_filter, padding=dilation, dilation=dilation).view(b, c, h, w)
        djj = F.conv2d(inputs.view(-1, 1, h, w), djj_filter, padding=dilation, dilation=dilation).view(b, c, h, w)

        det = dii * djj - dij * dij
        tr = dii + djj
        thld = (edge_thld + 1) ** 2 / edge_thld
        is_not_edge = torch.min(tr * tr / det <= thld, det > 0)

        return is_not_edge
