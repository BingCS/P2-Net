import os
import cv2
import open3d
import pickle
import numpy as np


def get_dist_list(feat1, feat2):

    cos_dist_mat = feat1 @ feat2.T
    norm1 = np.sum(feat1 * feat1, axis=1, keepdims=True)
    norm2 = np.sum(feat2 * feat2, axis=1, keepdims=True)
    dist_mat = norm1 - 2 * cos_dist_mat + norm2.transpose()
    nearest_dist = np.min(dist_mat, axis=1)
    nearest_idx = np.argmin(dist_mat, axis=1)

    return nearest_dist, nearest_idx


def compute_pose_error(kpts1, kpts2_3d_2, K, pose):
    def angle_error(R1, R2):
        cos = np.clip((np.trace(np.dot(np.linalg.inv(R1), R2)) - 1) / 2, a_min=None, a_max=1.0)
        return np.rad2deg(np.abs(np.arccos(cos)))

    kpts1 = kpts1.astype(np.float32).reshape((-1, 1, 2))
    kpts2_3d_2 = kpts2_3d_2.astype(np.float32).reshape((-1, 1, 3))

    success, R_vec, t, inliers = cv2.solvePnPRansac(
        kpts2_3d_2, kpts1, K, np.zeros(4), flags=cv2.SOLVEPNP_P3P,
        iterationsCount=1000, reprojectionError=3)
    if not success:
        return ValueError

    R, _ = cv2.Rodrigues(R_vec)
    t = t[:, 0]

    T_2to1 = np.linalg.inv(pose)
    error_t = np.linalg.norm(t - T_2to1[:3, 3])
    error_R = angle_error(R, T_2to1[:3, :3])
    return error_t, error_R, inliers[:-1, 0]


def get_matching_indices(anc_pts, pos_pts, coarse, fine):
    match_inds_coarse = []
    match_inds_fine = []
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
    match = bf_matcher.match(anc_pts, pos_pts)
    for match_val in match:
        if match_val.distance < coarse:
            match_inds_coarse.append([match_val.queryIdx, match_val.trainIdx])
            if match_val.distance < fine:
                match_inds_fine.append([match_val.queryIdx, match_val.trainIdx])
    return np.array(match_inds_coarse), np.array(match_inds_fine)

# root = '/Volumes/Bing_S/'
root = '/hy-tmp/datasets/'
modes = ['test', 'train']
src_scene_path = root + 'P2NET_7Scenes'
split_data_path = root + 'P2NET_7Scenes'
p2net_path = root + 'P2NET_7Scenes'

train_test_seq = {'7-scenes-chess': {'train': ['seq-01', 'seq-02', 'seq-04', 'seq-06'], 'test': ['seq-03', 'seq-05']},
                 '7-scenes-fire': {'train': ['seq-01', 'seq-02'], 'test': ['seq-03', 'seq-04']},
                 '7-scenes-heads': {'train': ['seq-02'], 'test': ['seq-01']},
                 '7-scenes-office': {'train': ['seq-01', 'seq-03', 'seq-04', 'seq-05', 'seq-08', 'seq-10'],
                            'test': ['seq-02', 'seq-06', 'seq-07', 'seq-09']},
                 '7-scenes-pumpkin': {'train': ['seq-02', 'seq-03', 'seq-06', 'seq-08'], 'test': ['seq-01', 'seq-07']},
                 '7-scenes-redkitchen': {'train': ['seq-01', 'seq-02', 'seq-05', 'seq-07', 'seq-08', 'seq-11', 'seq-13'],
                                'test': ['seq-03', 'seq-04', 'seq-06', 'seq-12', 'seq-14']},
                 '7-scenes-stairs': {'train': ['seq-02', 'seq-03', 'seq-05', 'seq-06'], 'test': ['seq-01', 'seq-04']}}

corrs_train = {}
corrs_train_filename = os.path.join(p2net_path, f'P2NET_train_0.010_corrs_7scenes.pkl')

pairs_train = {}
pairs_train_filename = os.path.join(p2net_path, f'P2NET_train_0.030_pairs_7scenes.pkl')

corrs_test = {}
corrs_test_filename = os.path.join(p2net_path, f'P2NET_test_0.010_corrs_7scenes.pkl')

pairs_test = {}
pairs_test_filename = os.path.join(p2net_path, f'P2NET_test_0.030_pairs_7scenes.pkl')
skip = 5

loop_list = np.arange(0, 5)
scene_list = sorted(os.listdir(src_scene_path))
for scene in scene_list[0:]:
    if scene == '7-scenes-stairs':
        frame_len = 500
    else:
        frame_len = 1000

    start_frame_list = np.arange(0, frame_len, skip)
    scene_path = os.path.join(src_scene_path, scene)
    K = np.loadtxt(os.path.join(scene_path, 'camera-intrinsics.txt'))

    seq_list = sorted(os.listdir(scene_path))[1:]
    for seq in seq_list:
        id_root = os.path.join(scene, seq)
        print(id_root)
        seq_path = os.path.join(scene_path, seq)

        for start_frame in start_frame_list:
            points_list = []
            pose_list = []
            depth_list = []
            ptcld_name_list = []
            select_frame_list = np.arange(start_frame, start_frame + skip)
            for frame_idx in select_frame_list:
                pos_ptcld_id = os.path.join(scene_path, seq, f'ptcld_{frame_idx}.pkl')
                depth = cv2.imread(os.path.join(scene_path, seq, f'frame-{frame_idx:06d}.depth.png'), -1)
                pose = np.loadtxt(os.path.join(scene_path, seq, f'frame-{frame_idx:06d}.pose.txt'))
                pose_list.append(pose)
                depth_list.append(depth)
                ptcld_name_list.append(pos_ptcld_id)

                ########################################################################################
                ########################################################################################
                ########################################################################################

                # ptlcd_gen_from depth
                pos_idx_all = np.where((depth > 0) & (depth < 65535))

                valid_h_idx = (pos_idx_all[0] >= 5) & (pos_idx_all[0] < 475)  # [True, False, ...]
                valid_w_idx = (pos_idx_all[1] >= 5) & (pos_idx_all[1] < 635)
                valid_idx = valid_h_idx & valid_w_idx

                valid_h = pos_idx_all[0][valid_idx]
                valid_w = pos_idx_all[1][valid_idx]
                valid_pos = np.vstack((valid_h, valid_w))
                valid_depth = (depth[valid_h, valid_w] / 1000.0)

                valid_uv = np.vstack((valid_w, valid_h))
                uv1_homo = np.vstack((valid_uv, np.ones(len(valid_h))))
                xy1_homo = np.matmul(np.linalg.inv(K), uv1_homo)  # (3, N')
                xyz1_homo = np.vstack(((np.expand_dims(valid_depth, axis=0) * xy1_homo), np.ones(len(valid_h))))
                xyz_homo = np.matmul(pose, xyz1_homo)
                xyz = xyz_homo[:3, :].transpose()
                points_list.append(xyz)

            submap_points = np.concatenate(points_list, axis=0)
            submap_pcd = open3d.geometry.PointCloud()
            submap_pcd.points = open3d.utility.Vector3dVector(submap_points)
            down_submap_pcd = open3d.geometry.PointCloud.voxel_down_sample(submap_pcd, voxel_size=0.015)

            down_submap_points = np.asarray(down_submap_pcd.points)

            for frame_id in select_frame_list:

                # ptlcd_crop_from submap
                frame_idy = frame_id - start_frame
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(down_submap_points)
                pcd.transform(np.linalg.inv(pose_list[frame_idy]))
                unaligned_pt = np.asarray(pcd.points).astype(np.float32)

                anc_xyz_homo = unaligned_pt.transpose()  # (3, kpt_n)
                anc_xy_homo = anc_xyz_homo / np.expand_dims(anc_xyz_homo[-1, :], axis=0)  # (3, kpt_n) / (1, kpt_n)
                anc_uv_tmp = np.matmul(K, anc_xy_homo)[0:2, :]  # (2, kpt_n)
                pos_tmp = np.vstack((anc_uv_tmp[1, :], anc_uv_tmp[0, :]))

                anc_valid_h_idx = (pos_tmp[0, :] >= 5) & (pos_tmp[0, :] < 475)  # [True, False, ...]
                anc_valid_w_idx = (pos_tmp[1, :] >= 5) & (pos_tmp[1, :] < 635)
                anc_valid_idx = anc_valid_h_idx & anc_valid_w_idx

                anc_valid_points = down_submap_points[anc_valid_idx]
                anc_valid_points_len = len(anc_valid_points)

                print(anc_valid_points_len)
                with open(ptcld_name_list[frame_idy], 'wb') as file:
                    pickle.dump(anc_valid_points, file)

                if anc_valid_points_len >= 3000 and anc_valid_points_len <= 120000:

                    filter_uv = np.around(anc_uv_tmp[:, anc_valid_idx])
                    anc_uv = np.unique(filter_uv, axis=1)
                    anc_pos = np.vstack((anc_uv[1, :], anc_uv[0, :])).astype(np.int32)

                    # ptlcd_gen_from depth
                    potential_depth = depth_list[frame_idy][anc_pos[0, :], anc_pos[1, :]]
                    depth_mask = np.where((potential_depth > 0) & (potential_depth < 65535))[0]
                    valid_pos = np.vstack((anc_pos[0, depth_mask], anc_pos[1, depth_mask]))
                    valid_depth = potential_depth[depth_mask] / 1000.0

                    valid_uv = np.vstack((anc_pos[1, depth_mask], anc_pos[0, depth_mask]))
                    uv1_homo = np.vstack((valid_uv, np.ones(len(valid_depth))))
                    xy1_homo = np.matmul(np.linalg.inv(K), uv1_homo)  # (3, N')
                    xyz1_homo = np.vstack(((np.expand_dims(valid_depth, axis=0) * xy1_homo), np.ones(len(valid_depth))))
                    xyz_homo = np.matmul(pose_list[frame_idy], xyz1_homo)
                    pos_points = xyz_homo[:3, :].transpose()

                    ########################################################################################
                    ########################################################################################
                    ########################################################################################

                    matched_pcd_coarse, matched_pcd_fine = get_matching_indices(pos_points.astype(np.float32), anc_valid_points.astype(np.float32), 0.015, 0.001)
                    used_len = len(np.unique(matched_pcd_coarse[:, 1]))
                    anc_pcd_keypts, unique_mask = np.unique(matched_pcd_fine[:, 1], return_index=True)
                    pos_pcd_keypts = matched_pcd_fine[:, 0][unique_mask]
                    corr_len = len(pos_pcd_keypts)

                    corr_pos_idx = valid_pos[:, pos_pcd_keypts].transpose()
                    corr_ptcld_idx = anc_pcd_keypts
                    overlap = used_len / anc_valid_points_len

                    if corr_len >= 128:
                        key_id = os.path.join(id_root, f'ptcld_{frame_id}.pkl')
                        if seq in train_test_seq[scene]['train']:
                            corrs_train[key_id] = np.concatenate(
                                (np.expand_dims(corr_ptcld_idx, axis=1), corr_pos_idx), axis=1).astype(np.float32)
                            pairs_train[key_id] = [corr_len, used_len, anc_valid_points_len, overlap]
                        else:
                            corrs_test[key_id] = np.concatenate(
                                (np.expand_dims(corr_ptcld_idx, axis=1), corr_pos_idx), axis=1).astype(np.float32)
                            pairs_test[key_id] = [corr_len, used_len, anc_valid_points_len, overlap]

    # ###update corrs after process of each scene.
    with open(corrs_train_filename, 'wb') as corrs_file_train:
        pickle.dump(corrs_train, corrs_file_train)

    with open(pairs_train_filename, 'wb') as pairs_file_train:
        pickle.dump(pairs_train, pairs_file_train)

    with open(corrs_test_filename, 'wb') as corrs_file_test:
        pickle.dump(corrs_test, corrs_file_test)

    with open(pairs_test_filename, 'wb') as pairs_file_test:
        pickle.dump(pairs_test, pairs_file_test)

    print(f"finished generation of {scene} ")
