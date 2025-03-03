# Basic libs
import numpy as np
import cv2
import random
import math

# OS functions and io
import os
from os.path import join
import pickle

# Dataset parent class
from data.common import PointCloudDataset
from data.tools.geom import photometric_augmentation
# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def rotate(points, num_axis=1):
    if num_axis == 1:
        theta = np.random.rand() * 2 * np.pi
        axis = np.random.randint(3)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, -s], [s, c, -s], [s, s, c]], dtype=np.float32)
        R[:, axis] = 0
        R[axis, :] = 0
        R[axis, axis] = 1
        points = np.matmul(points, R)
    elif num_axis == 3:
        for axis in [0, 1, 2]:
            theta = np.random.rand() * 2 * np.pi
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, -s], [s, c, -s], [s, s, c]], dtype=np.float32)
            R[:, axis] = 0
            R[axis, :] = 0
            R[axis, axis] = 1
            points = np.matmul(points, R)
    else:
        exit(-1)
    return points


def fix_img_trans(img, stats):
    img = (img - stats[0]) / (stats[1] + 1e-5)
    img = img.transpose(0, 3, 1, 2)
    img = np.float32(img)  # [B, H, W, C]
    return img


def online_img_trans(img):
    img_mean = np.mean(img, axis=(0, 1))
    std = np.std(img, axis=(0, 1))
    img = (img - img_mean) / (std + 1e-5)
    img = img.transpose(2, 0, 1)
    img = np.float32(img)  # [H, W, C]
    return img

# ----------------------------------------------------------------------------------------------------------------------
#
#           Class Definition
#       \***************/
#


class P2NETMatchDataset(PointCloudDataset):
    """
    Class to handle P2PMatch dataset for dense keypoint detection and feature description task.
    """

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, opt, mode='train', voxel_size=0.015):
        PointCloudDataset.__init__(self, opt)
        self.mode = mode
        self.downsample = voxel_size
        self.opt = opt

        self.keypts = {}
        self.corrs_id_list = []
        self.batched_corrs_list = []
        self.pairs_num = 0
        self.steps = 0

        self.prepare_P2NET_ply(mode=self.mode)
        self.reset_batch_list(self.opt.batchsize)

        print("Num {:s} steps : {:d}".format(mode, self.steps))

    def __len__(self):
        """
        Return the length of data here
        """
        return self.steps

    def __getitem__(self, index):

        # Initiate concatenation lists
        anc_points_list = []
        pos_images_list = []
        anc_keypts_list = []
        pos_keypts_list = []
        backup_points_list = []
        valid_depth_mask_list = []
        input_ply_id = []

        batched_corrs_id = self.batched_corrs_list[index]
        for anc_id in batched_corrs_id:
            frame_name = anc_id.split('.')[0]
            num_str = frame_name.split('_')[-1]
    
            num_str_padded = num_str.zfill(6)
    
            pos_id = frame_name.replace('ptcld_' + num_str, 'frame-' + num_str_padded)


            # read image and point cloud
            with open(join(self.opt.img_dir, anc_id), 'rb') as f:
                anc_points = pickle.load(f).astype(np.float32)
                backup_points = anc_points

            img_filename = join(self.opt.img_dir, f'{pos_id}.color.png')
            if self.opt.img_mode == 'gray':
                pos_images = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
            elif self.opt.img_mode == 'color':
                pos_images = cv2.imread(img_filename)
            else:
                print("Wrong image type!")
                raise ValueError

            # online image norm
            if self.opt.img_trans_mode == 'online':
                if self.opt.img_mode == 'gray':
                    pos_images = np.expand_dims(pos_images, axis=-1)
                pos_images = online_img_trans(pos_images)

            depth_file_name = img_filename.replace('img', 'depth')
            depth_img = cv2.imread(depth_file_name, -1)
            valid_depth_mask = (depth_img > 0) & (depth_img < 65535)

            anc_keypts = self.keypts[anc_id][:, 0]
            pos_keypts = self.keypts[anc_id][:, 1:]

            if min(len(anc_keypts), len(pos_keypts)) > self.opt.kpt_n:
                selected_ind = np.random.choice(min(len(anc_keypts), len(pos_keypts)), self.opt.kpt_n, replace=False)
                anc_keypts = anc_keypts[selected_ind]
                pos_keypts = pos_keypts[selected_ind].transpose()
            else:
                anc_keypts = anc_keypts
                pos_keypts = pos_keypts.transpose()

            anc_points_list += [anc_points]
            pos_images_list += [pos_images]
            backup_points_list += [backup_points]
            valid_depth_mask_list += [valid_depth_mask]
            anc_keypts_list += [anc_keypts.astype(np.int64)]
            pos_keypts_list += [pos_keypts.astype(np.float32)]
            input_ply_id += [np.array([anc_id, pos_id])]

        # Add data to current batch
        input_pos_images = np.array(pos_images_list)
        input_anc_points = np.concatenate(anc_points_list, axis=0)
        input_backup_points = np.concatenate(backup_points_list, axis=0)
        input_stack_lengths = np.array([tp.shape[0] for tp in anc_points_list], dtype=np.int32)
        input_stacked_features = np.ones_like(input_anc_points[:, :1], dtype=np.float32)

        # full_image_norm
        if self.opt.img_trans_mode == 'fix':
            if self.opt.img_mode == 'gray':
                input_pos_images = np.expand_dims(input_pos_images, axis=-1)
            input_pos_images = fix_img_trans(input_pos_images, self.scene_stats)

        input_list = [input_pos_images, valid_depth_mask_list]
        input_list += self.descriptor_inputs(input_anc_points, input_stacked_features, input_stack_lengths, self.mode)
        input_list += [input_stack_lengths, anc_keypts_list, pos_keypts_list, input_backup_points, np.array(input_ply_id)]

        return [self.opt.num_layers] + input_list

    def prepare_P2NET_ply(self, mode='train'):
        """
        Load pre-generated point cloud, keypoint correspondence(the indices) to save time.
        Construct the self.anc_to_pos dictionary.
        """
        print(f'\nPreparing {mode} ply files')
        with open(join(self.opt.data_dir, f'P2NET_{mode}_0.010_corrs_7scenes.pkl'), 'rb') as f:
            self.keypts = pickle.load(f)

        self.corrs_id_list = list(self.keypts.keys())
        self.pairs_num = len(self.corrs_id_list)

        print("Num {:s} paris : {:d}".format(mode, self.pairs_num))

    def reset_batch_list(self, batchsize):
        """
        Reset the correspondence idx list in batch mode. Split the self.corrs_id_list into 
        groups with equal length of batchsize. 
        """
        print(f"Construct {self.mode} batched correspondence list...")
        # random.shuffle(self.corrs_id_list)
        self.batched_corrs_list = [self.corrs_id_list[i : i + batchsize] for i in range(0, len(self.corrs_id_list), batchsize)]
        while len(self.batched_corrs_list[-1]) < batchsize:
            self.batched_corrs_list[-1].append(self.batched_corrs_list[-1][-1])
        self.steps = len(self.batched_corrs_list)
