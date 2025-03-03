import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.libs.kpconv_lib import block_decider


class KPFCNN(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        return

    def forward(self, batch, opt):

        # Get input features
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)

        neighbors = batch.neighbors[0]  # [n_points, n_neighbors[0]]
        pcd_length = batch.stack_lengths  # [len1, len2, ...]
        tmp_x = x

        # add a fake point in the last row for shadow neighbors
        shadow_features = torch.zeros_like(x[:1, :])
        x = torch.cat([x, shadow_features], dim=0)  # [total_points_number + 1, d]
        shadow_neighbor = torch.ones_like(neighbors[:1, :]) * torch.sum(pcd_length)
        neighbors = torch.cat([neighbors, shadow_neighbor], dim=0)

        descs_batch = {'pred':[], 'rand':[]}
        kpts_batch = {'pred':[], 'rand':[]}
        scores_batch = {'pred':[], 'rand':[]}
        valid_pos = []
        valid_feat = []
        valid_score = []
        for kpts_indices in batch.anc_keypts_inds:
            if opt.isTrain:
                max_per_sample = torch.max(x[:-1])
                x = x / (max_per_sample + 1e-6)

                score_map = self.peakiness_score(x, neighbors, kpts_indices, opt)
                valid_pos.append(batch.backup_points[kpts_indices].t())
                valid_feat.append(tmp_x[kpts_indices].t())  # [kpt_n, d]
                valid_score.append(score_map)
                return valid_feat, valid_pos, valid_score

            else:
                score_map = self.point_hard_selection(x, neighbors, opt)
                # no keypoints index supplied, need to drop last line
                score = score_map[:-1]  # [total_points_number]
                sample = torch.argsort(score, descending=True)[:opt.kpt_n]
                print("d3 keypts:", len(torch.nonzero(score_map)), len(batch.backup_points))

                # pred selection
                pred_kpts = batch.backup_points[sample]
                pred_score = score[sample]
                pred_descs = F.normalize(tmp_x[sample], dim=-1)

                # rand selection
                rand_sample = torch.randperm(len(batch.backup_points))[:opt.kpt_n]
                rand_kpts = batch.backup_points[rand_sample]
                rand_score = score[rand_sample]
                rand_descs = F.normalize(tmp_x[rand_sample], dim=-1)

                kpts_batch['pred'].append(pred_kpts.cpu().numpy())
                descs_batch['pred'].append(pred_descs.cpu().numpy())
                scores_batch['pred'].append(pred_score.cpu().numpy())
                kpts_batch['rand'].append(rand_kpts.cpu().numpy())
                descs_batch['rand'].append(rand_descs.cpu().numpy())
                scores_batch['rand'].append(rand_score.cpu().numpy())
                return descs_batch, kpts_batch, scores_batch


    @staticmethod
    def peakiness_score(features, neighbors, pos, opt):
        # get valid_neighbors first
        neighbors = neighbors[pos, :]

        # local max score (saliency score)
        neighbor_features = features[neighbors, :]  # [n_points, n_neighbors, d]
        neighbor_features_sum = torch.sum(neighbor_features, dim=-1)  # [n_points, n_neighbors]
        neighbor_num = torch.sum(neighbor_features_sum != 0, dim=-1, keepdim=True)
        neighbor_num = torch.clamp_min(neighbor_num, min=1)
        mean_features = torch.sum(neighbor_features, dim=1) / neighbor_num.to(torch.float32)  # [n_points, d]

        features = features[pos]
        alpha = F.softplus(features - mean_features)  # [n_points, d]
        if opt.beta_mode == 'softplus':
            beta = F.softplus(features - torch.mean(features, dim=1, keepdim=True))
        else:
            beta = features / (torch.max(features, dim=1, keepdim=True)[0] + 1e-6)

        score_vol = alpha * beta
        score_map = torch.max(score_vol, dim=1, keepdim=True)[0]  # [n_points, 1]

        return score_map.squeeze(-1)

    @staticmethod
    def point_hard_selection(features, neighbors, opt):
        # local max score (saliency score)
        neighbor_features = features[neighbors, :]  # [n_points, n_neighbors, d]
        neighbor_features_sum = torch.sum(neighbor_features, dim=-1)  # [n_points, n_neighbors]
        neighbor_num = torch.sum(neighbor_features_sum != 0, dim=-1, keepdim=True)
        neighbor_num = torch.clamp_min(neighbor_num, min=1)
        mean_features = torch.sum(neighbor_features, dim=1) / neighbor_num.to(torch.float32)  # [n_points, d]

        alpha = F.softplus(features - mean_features)  # [n_points, d]
        if opt.beta_mode == 'softplus':
            beta = F.softplus(features - torch.mean(features, dim=1, keepdim=True))
        else:
            beta = features / (torch.max(features, dim=1, keepdim=True)[0] + 1e-6)

        score_vol = alpha * beta
        score_map = torch.max(score_vol, dim=1, keepdim=True)[0]  # [n_points, 1]
        # print("raw score map(alpha x beta) and its max, min, median:", score_map.size(), score_map, torch.max(score_map), torch.min(score_map), torch.median(score_map))

        """
        # hard selection (used during test)
        local_max = torch.max(neighbor_features, dim=1)[0]
        is_local_max = torch.eq(features, local_max)

        depth_wise_max = torch.max(features, dim=1, keepdim=True)[0]
        is_depth_wise_max = torch.eq(features, depth_wise_max)

        detected = torch.max(is_local_max & is_depth_wise_max, dim=1, keepdim=True)[0]
        score_map = score_map * detected
        """
        local_max = torch.max(neighbor_features, dim=1)[0]
        is_local_max = torch.eq(features, local_max)
        # print("num of local max :", torch.sum(is_local_max, dtype=torch.int32))
        detected = torch.max(is_local_max.to(torch.float32), dim=1, keepdim=True)[0]
        score_map = score_map * detected

        # print("detected score map:", score_map)

        return score_map.squeeze(-1)

    def finetune(self, finetune_layers, finetune_blocks='decoder'):
        for param in self.parameters():
            param.requires_grad = False
        n = 0
        if finetune_blocks == 'decoder':
            for param in self.decoder_blocks[finetune_layers].parameters():
                param.requires_grad = True
                n += 1
        else:
            for param in self.encoder_blocks[finetune_layers].parameters():
                param.requires_grad = True
                n += 1
        if n == 0:
            raise NotImplementedError

        print(f"D3 finetune layers : {finetune_blocks} {finetune_layers} ")
