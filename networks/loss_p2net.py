import torch
import torch.nn.functional as F
from networks.libs.base_lib import get_dist_mat


def p2net_criterion(d2_out, d3_out, pos1, opt, steps, loss_mode):

    ####################################
    #    Get scores and descriptors
    ####################################
    dense_feat_map1, score_map1 = d2_out

    feat2 = d3_out[0]
    pos2 = d3_out[1]
    score2 = d3_out[2]

    if opt.loss_type == 'CIRCLE':
        dist_type = 'cosine_dist'
    elif opt.loss_type.find('HARD') >= 0:
        dist_type = 'euclidean_dist'
    else:
        raise NotImplementedError

    total_d_pos = None
    total_d_neg = None
    total_d_neg_row = None
    total_d_neg_col = None
    total_loss = None
    total_det_loss = None
    total_accuracy = None
    for i in range(opt.batchsize):
        valid_pos1 = pos1[i]
        valid_pos2 = pos2[i]
        kpt_n = valid_pos1.size(-1)

        valid_feat1 = dense_feat_map1[i][:, valid_pos1[0, :].long(), valid_pos1[1, :].long()]  # full size, extract directly
        valid_score1 = score_map1[i][:, valid_pos1[0, :].long(), valid_pos1[1, :].long()]  # full size, extract directly

        valid_feat2 = feat2[i]
        valid_score2 = score2[i]

        valid_feat1 = F.normalize(valid_feat1, dim=0)  # [128, kpt_n]
        valid_feat2 = F.normalize(valid_feat2, dim=0)  # [128, kpt_n]

        ####################################
        #    Compute dist_mat and mask
        ####################################
        radius_mask_row = get_dist_mat(valid_pos2, valid_pos2, "euclidean_dist_no_norm")
        radius_mask_row = torch.lt(radius_mask_row, opt.safe_radius)
        radius_mask_col = get_dist_mat(valid_pos1, valid_pos1, "euclidean_dist_no_norm")
        radius_mask_col = torch.lt(radius_mask_col, opt.safe_pixel)
        radius_mask_row = radius_mask_row.to(torch.float32) - torch.eye(kpt_n, device=opt.device)
        radius_mask_col = radius_mask_col.to(torch.float32) - torch.eye(kpt_n, device=opt.device)

        dist_mat = get_dist_mat(valid_feat1.unsqueeze(0), valid_feat2.unsqueeze(0), dist_type)
        pos_vec = torch.diagonal(dist_mat, offset=0, dim1=-2, dim2=-1)

        ####################################
        #  detector loss & descriptor loss
        ####################################
        if loss_mode == 'Joint':
            det_loss = torch.zeros(1)
            desc_loss = descriptor_loss(dist_mat, pos_vec, radius_mask_row, radius_mask_col, kpt_n,
                                        opt.loss_type, opt.device, opt)

            kpt_weight = valid_score1 * valid_score2
            desc_loss = desc_loss * kpt_weight
            loss = torch.sum(desc_loss) / (torch.sum(kpt_weight) + 1e-6)

        elif loss_mode == 'Split':
            det_loss = detector_loss(dist_mat, pos_vec, radius_mask_row, radius_mask_col, kpt_n,
                                    dist_type, opt.device)
            desc_loss = descriptor_loss(dist_mat, pos_vec, radius_mask_row, radius_mask_col, kpt_n,
                                        opt.loss_type, opt.device, opt)

            desc_loss = torch.mean(desc_loss)
            det_loss = torch.mean(det_loss * (valid_score1 + valid_score2 + 1e-6))
            loss = desc_loss + det_loss

        elif loss_mode == 'Single':
            det_loss = torch.zeros(1)
            desc_loss = descriptor_loss(dist_mat, pos_vec, radius_mask_row, radius_mask_col, kpt_n,
                                        opt.loss_type, opt.device, opt)

            desc_loss = torch.mean(desc_loss)
            loss = desc_loss

        elif loss_mode == 'Ours':
            ori_det_loss = detector_loss(dist_mat, pos_vec, radius_mask_row, radius_mask_col, kpt_n,
                                    dist_type, opt.device)
            desc_loss = descriptor_loss(dist_mat, pos_vec, radius_mask_row, radius_mask_col, kpt_n,
                                        opt.loss_type, opt.device, opt)

            kpt_weight = valid_score1 * valid_score2
            det_loss = torch.sum(ori_det_loss * kpt_weight) / (torch.sum(kpt_weight) + 1e-6)
            desc_loss = torch.mean(desc_loss)
            loss = det_loss + desc_loss

        else:
            raise NotImplementedError

        #####################################
        #     Compute accuracy and d_neg
        #####################################
        if dist_type == 'cosine_dist':
            err_row = dist_mat - pos_vec.unsqueeze(-1)
            err_col = dist_mat - pos_vec.unsqueeze(-2)

            d_pos = torch.mean(pos_vec)
            d_neg_row = torch.mean(torch.max(dist_mat - 1e5 * torch.eye(kpt_n, device=opt.device) - 1e5 * radius_mask_row, dim=-1)[0])
            d_neg_col = torch.mean(torch.max(dist_mat - 1e5 * torch.eye(kpt_n, device=opt.device) - 1e5 * radius_mask_col, dim=-2)[0])
            d_neg = (d_neg_row + d_neg_col) / 2

        elif dist_type == 'euclidean_dist' or dist_type == 'euclidean_dist_no_norm':
            err_row = pos_vec.unsqueeze(-1) - dist_mat
            err_col = pos_vec.unsqueeze(-2) - dist_mat

            d_pos = torch.mean(pos_vec)
            d_neg_row = torch.mean(torch.min(dist_mat + 1e5 * torch.eye(kpt_n, device=opt.device) + 1e5 * radius_mask_row, dim=-1)[0])
            d_neg_col = torch.mean(torch.min(dist_mat + 1e5 * torch.eye(kpt_n, device=opt.device) + 1e5 * radius_mask_col, dim=-2)[0])
            d_neg = (d_neg_row + d_neg_col) / 2

        else:
            raise NotImplementedError

        err_row = torch.sum(F.relu(err_row), dim=-1)
        err_col = torch.sum(F.relu(err_col), dim=-2)

        cnt_err_row = len(torch.nonzero(err_row))
        cnt_err_col = len(torch.nonzero(err_col))
        tot_err = cnt_err_row + cnt_err_col
        accuracy = 1. - tot_err / kpt_n / 2.

        if i == 0:
            total_d_pos = d_pos
            total_d_neg = d_neg
            total_d_neg_row = d_neg_row
            total_d_neg_col = d_neg_col
            total_loss = loss
            total_det_loss = det_loss
            total_accuracy = accuracy
        else:
            total_d_pos += d_pos
            total_d_neg += d_neg
            total_d_neg_row += d_neg_row
            total_d_neg_col += d_neg_col
            total_loss += loss
            total_det_loss += det_loss
            total_accuracy += accuracy

    mean_loss = total_loss / opt.batchsize
    mean_det_loss = total_det_loss / opt.batchsize
    mean_accuracy = total_accuracy / opt.batchsize
    mean_d_pos = total_d_pos / opt.batchsize
    mean_d_neg = total_d_neg / opt.batchsize
    mean_d_neg_row = total_d_neg_row / opt.batchsize
    mean_d_neg_col = total_d_neg_col / opt.batchsize

    return mean_loss, mean_det_loss, mean_accuracy, mean_d_pos, mean_d_neg, mean_d_neg_row, mean_d_neg_col


def detector_loss(dist_mat, pos_vec, radius_mask_row, radius_mask_col, kpt_n, dist_type, device):

    if dist_type == 'cosine_dist':
        dist_mat_without_min_on_diag = dist_mat - 1e5 * torch.unsqueeze(torch.eye(kpt_n, device=device), 0)
        hard_neg_dist_row = dist_mat_without_min_on_diag  #  - 1e5 * radius_mask_row
        hard_neg_dist_col = dist_mat_without_min_on_diag  #  - 1e5 * radius_mask_col
        hard_neg_dist_row = torch.max(hard_neg_dist_row, dim=-1)[0]
        hard_neg_dist_col = torch.max(hard_neg_dist_col, dim=-2)[0]

        det_loss_row = hard_neg_dist_row - pos_vec
        det_loss_col = hard_neg_dist_col - pos_vec
        det_loss = (det_loss_row + det_loss_col) / 2

    elif dist_type == 'euclidean_dist':
        dist_mat_without_min_on_diag = dist_mat + 1e5 * torch.unsqueeze(torch.eye(kpt_n, device=device), 0)
        hard_neg_dist_row = dist_mat_without_min_on_diag + 1e5 * radius_mask_row
        hard_neg_dist_col = dist_mat_without_min_on_diag + 1e5 * radius_mask_col
        hard_neg_dist_row = torch.min(hard_neg_dist_row, dim=-1)[0]
        hard_neg_dist_col = torch.min(hard_neg_dist_col, dim=-2)[0]

        det_loss_row = pos_vec - hard_neg_dist_row
        det_loss_col = pos_vec - hard_neg_dist_col
        det_loss = (det_loss_row + det_loss_col) / 2

    else:
        raise NotImplementedError

    return det_loss


def descriptor_loss(dist_mat, pos_vec, radius_mask_row, radius_mask_col, kpt_n, loss_type, device, opt):

    if loss_type == 'HARD_CONTRASTIVE':
        # using (euclidean distance)
        pos_margin = opt.pos_margin
        neg_margin = opt.neg_margin

        dist_mat_without_min_on_diag = dist_mat + 1e5 * torch.unsqueeze(torch.eye(kpt_n, device=device), 0)
        hard_neg_dist_row = dist_mat_without_min_on_diag + 1e5 * radius_mask_row
        hard_neg_dist_col = dist_mat_without_min_on_diag + 1e5 * radius_mask_col
        hard_neg_dist_row = torch.min(hard_neg_dist_row, dim=-1)[0]
        hard_neg_dist_col = torch.min(hard_neg_dist_col, dim=-2)[0]

        pos_loss = F.relu(pos_vec - pos_margin)
        loss_row = pos_loss + F.relu(-hard_neg_dist_row + neg_margin)
        loss_col = pos_loss + F.relu(-hard_neg_dist_col + neg_margin)
        desc_loss = (loss_row + loss_col) / 2

    elif loss_type == 'HARD_TRIPLET':
        # using (euclidean distance)
        margin = 0.5

        dist_mat_without_min_on_diag = dist_mat + 1e5 * torch.unsqueeze(torch.eye(kpt_n, device=device), 0)
        hard_neg_dist_row = dist_mat_without_min_on_diag + 1e5 * radius_mask_row
        hard_neg_dist_col = dist_mat_without_min_on_diag + 1e5 * radius_mask_col
        hard_neg_dist_row = torch.min(hard_neg_dist_row, dim=-1)[0]
        hard_neg_dist_col = torch.min(hard_neg_dist_col, dim=-2)[0]

        loss_row = F.relu(pos_vec - hard_neg_dist_row + margin)
        loss_col = F.relu(pos_vec - hard_neg_dist_col + margin)
        desc_loss = (loss_row + loss_col) / 2

    elif loss_type == 'CIRCLE':
        # using (cosine distance)
        log_scale = 10
        m = 0.2

        neg_mask_row = torch.unsqueeze(torch.eye(kpt_n, device=device), 0)
        neg_mask_row += radius_mask_row
        neg_mask_col = torch.unsqueeze(torch.eye(kpt_n, device=device), 0)
        neg_mask_col += radius_mask_col

        pos_margin = -m + 1
        neg_margin = m
        pos_optimal = m + 1
        neg_optimal = -m

        neg_mat_row = dist_mat - 128 * neg_mask_row
        neg_mat_col = dist_mat - 128 * neg_mask_col

        lse_positive = torch.logsumexp(-log_scale * (pos_vec[..., None] - pos_margin) *
                                       torch.clamp_min(-pos_vec[..., None] + pos_optimal, min=1e-6).detach(), -1)
        lse_negative_row = torch.logsumexp(log_scale * (neg_mat_row - neg_margin) *
                                           torch.clamp_min(neg_mat_row - neg_optimal, min=1e-6).detach(), -1)
        lse_negative_col = torch.logsumexp(log_scale * (neg_mat_col - neg_margin) *
                                           torch.clamp_min(neg_mat_col - neg_optimal, min=1e-6).detach(), -2)

        loss_row = F.softplus(lse_positive + lse_negative_row) / log_scale
        loss_col = F.softplus(lse_positive + lse_negative_col) / log_scale
        desc_loss = (loss_row + loss_col) / 2

    else:
        raise NotImplementedError

    return desc_loss
