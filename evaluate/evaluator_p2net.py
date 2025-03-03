import cv2
import open3d
import numpy as np
import torch


class Evaluator(object):
    def __init__(self, opt):
        self.mutual_check = True
        self.r_thres = opt.r_thres
        self.t_thres = opt.t_thres
        self.err_thld_pixel = opt.err_thld_pixel
        self.err_thld_point = opt.err_thld_point
        self.err_thld_covisible = opt.err_thld_covisible
        self.stats = {
            'all_eval_stats': np.array((0, 0, 0, 0, 0, 0), np.float32)
        }
        print("err_thld_pixel: ", self.err_thld_pixel)
        print("err_thld_point: ", self.err_thld_point)
        print("err_thld_covisible: ", self.err_thld_covisible)

    def get_matching_indices(self, anc_pts, pos_pts):
        match_inds = []
        bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
        match = bf_matcher.match(anc_pts, pos_pts)
        for match_val in match:
            if match_val.distance < self.err_thld_covisible:
                match_inds.append([match_val.queryIdx, match_val.trainIdx])
        return np.array(match_inds)

    def points_trans(self, uv, depth, K, pose, device=None):
        # ptlcd_gen_from depth
        if device is not None:
            valid_depth = (depth[uv[1, :], uv[0, :]] / 1000.0)
            uv1_homo = torch.cat((uv.to(torch.float32), torch.ones(len(valid_depth), device=device).unsqueeze(0)))
            xy1_homo = torch.matmul(torch.inverse(K), uv1_homo)  # (3, N')
            xyz1_homo = torch.cat(((torch.unsqueeze(valid_depth, dim=0) * xy1_homo), torch.ones(len(valid_depth), device=device).unsqueeze(0)))
            xyz_homo = torch.matmul(pose, xyz1_homo)
            pos_points = xyz_homo[:3, :].transpose(1, 0).to(torch.float32)
        else:
            valid_depth = (depth[uv[1, :], uv[0, :]] / 1000.0)
            uv1_homo = np.vstack((uv, np.ones(len(valid_depth))))
            xy1_homo = np.matmul(np.linalg.inv(K), uv1_homo)  # (3, N')
            xyz1_homo = np.vstack(((np.expand_dims(valid_depth, axis=0) * xy1_homo), np.ones(len(valid_depth))))
            xyz_homo = np.matmul(pose, xyz1_homo)
            pos_points = xyz_homo[:3, :].transpose().astype(np.float32)
        return pos_points

    def pixel_trans(self, points, K, pose):
        # ptlcd_gen_from depth
        anc_pcd = open3d.geometry.PointCloud()
        anc_pcd.points = open3d.utility.Vector3dVector(points)
        anc_pcd.transform(np.linalg.inv(pose))
        unaligned_pt = np.asarray(anc_pcd.points)

        anc_xyz_homo = unaligned_pt.transpose()  # (3, kpt_n)
        anc_xy_homo = anc_xyz_homo / np.expand_dims(anc_xyz_homo[-1, :], axis=0)  # (3, kpt_n) / (1, kpt_n)
        anc_uv = np.matmul(K, anc_xy_homo)[0:2, :].transpose()  # (kpt_n, 2)
        return anc_uv

    def feature_matcher(self, descriptors_a, descriptors_b, device):
        sim = descriptors_a @ descriptors_b.t()
        nn12 = torch.max(sim, dim=1)[1]
        nn21 = torch.max(sim, dim=0)[1]
        ids1 = torch.arange(0, sim.shape[0], device=device)
        mask = (ids1 == nn21[nn12])
        matches = torch.stack([ids1[mask], nn12[mask]])

        return matches.t()

    def get_covisible_mask(self, ref_uv, test_points, depth, K, pose):
        ref_points = self.points_trans(ref_uv.transpose().astype(np.int32), depth, K, pose)
        mutual_mask = self.get_matching_indices(test_points, ref_points)
        test_mask = mutual_mask[:, 0]
        return test_mask

    def get_inlier_matches(self, ref_coord, test_coord, matches, depth, K, pose, device):

        irs, ins = [], []

        p_ref_coord = ref_coord[matches[:, 0]]
        p_test_coord = test_coord[matches[:, 1]]

        proj_p_ref_coord = self.points_trans(p_ref_coord.transpose(1, 0).to(torch.int64), depth, K, pose, device)
        dist = torch.sqrt(torch.sum(torch.pow((proj_p_ref_coord - p_test_coord), 2), dim=-1))


        dist_np = dist.cpu().numpy

        ins.append(np.sum(dist_np() < self.err_thld_point))
        irs.append(np.mean(dist_np() < self.err_thld_point))


        return np.array(ins), np.array(irs)

    def get_gt_matches(self, ref_coord, test_coord, depth, K, pose, device):
        proj_ref_coord = self.points_trans(ref_coord.transpose(1, 0).to(torch.int64), depth, K, pose, device)

        pt0 = torch.unsqueeze(proj_ref_coord, dim=1)
        pt1 = torch.unsqueeze(test_coord, dim=0)
        norm = torch.norm(pt0 - pt1, dim=2)

        min_dist0 = torch.min(norm, dim=1)[0]
        min_dist0_idx = torch.argmin(norm, dim=1)
        
        min_dist1 = torch.min(norm, dim=0)[0]
        min_dist1_idx = torch.argmin(norm, dim=0)

        # repeat num
        rep_min_dist1_mask = (min_dist1 <= 0.020)
        rep_min_dist0_mask = (min_dist0 <= 0.020)

        # gt num
        gt_min_dist1_mask = (min_dist1 <= self.err_thld_point)
        gt_min_dist0_mask = (min_dist0 <= self.err_thld_point)

        repeat_num0 = torch.sum(rep_min_dist0_mask)
        repeat_num1 = torch.sum(rep_min_dist1_mask)
        repeat_num = min(repeat_num0, repeat_num1)

        gt_num0 = len(torch.unique(min_dist0_idx[gt_min_dist0_mask]))
        gt_num1 = len(torch.unique(min_dist1_idx[gt_min_dist1_mask]))
        gt_num = min(gt_num0, gt_num1)
        return repeat_num, gt_num
    def calculate_inlier_rate(self, ref_coord, test_coord, matches, depth, K, pose, gt_num, device):
        # 计算内点
        p_ref_coord = ref_coord[matches[:, 0]]
        p_test_coord = test_coord[matches[:, 1]]

        proj_p_ref_coord = self.points_trans(p_ref_coord.transpose(1, 0).to(torch.int64), depth, K, pose, device)
        dist = torch.sqrt(torch.sum(torch.pow((proj_p_ref_coord - p_test_coord), 2), dim=-1))

        # 内点掩码
        mask = (dist <= self.err_thld_point)

        # 计算内点率
        inlier_rate = torch.sum(mask).item() / gt_num if gt_num > 0 else 0.0
        return inlier_rate
    def compute_pose_error(self, ref_coord, test_coord, matches, depth, K, pose):
        def angle_error(R1, R2):
            cos = np.clip((np.trace(np.dot(np.linalg.inv(R1), R2)) - 1) / 2, a_min=None, a_max=1.0)
            return np.rad2deg(np.abs(np.arccos(cos)))

        def rel_distance(T1, T2):
            R1 = T1[:3, :3]
            R2 = T2[:3, :3]
            t1 = T1[:3, 3]
            t2 = T2[:3, 3]
            d = np.dot(R1.T, t1) - np.dot(R2.T, t2)
            return np.linalg.norm(d)

        p_ref_coord = ref_coord[matches[:, 0]]
        p_test_coord = test_coord[matches[:, 1]]
        pred_pose = np.eye(4)

        uv = p_ref_coord.astype(np.float32).reshape((-1, 1, 2))
        points = p_test_coord.astype(np.float32).reshape((-1, 1, 3))

        reproj_error = 3
        iterationcount = 5000
        #1print('iter, reproj_error:', iterationcount, reproj_error)
        success, R_vec, t, inliers = cv2.solvePnPRansac(
            points, uv, K, np.zeros(4), iterationsCount=iterationcount, reprojectionError=reproj_error)

        R, _ = cv2.Rodrigues(R_vec)
        pred_pose[:3, 3] = t[:, 0]
        pred_pose[:3, :3] = R

        T_2to1 = np.linalg.inv(pose)
        error_t = round(np.linalg.norm(t[:, 0] - T_2to1[:3, 3]), 3)
        # error_t2 = rel_distance(np.linalg.inv(pred_pose), pose)
        # error_t3 = rel_distance(pred_pose, np.linalg.inv(pose))
        # print(error_t, error_t2, error_t3)
        error_R = angle_error(R, T_2to1[:3, :3])
        error = [error_t, error_R]

        real_warped_corners = self.points_trans(p_ref_coord.transpose().astype(np.int32), depth, K, pose)
        warped_corners = self.points_trans(p_ref_coord.transpose().astype(np.int32), depth, K, np.linalg.inv(pred_pose))
        mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
        correctness = float(mean_dist <= 0.05)
        if error_R < self.r_thres and error_t < self.t_thres:
            rrs = 1
        else:
            rrs = 0

        return error, correctness, mean_dist, rrs

    def print_stats(self, key):
        avg_stats = self.stats[key] / max(self.stats[key][0], 1)
        print('----------%s----------' % key)
        print('avg_rep', avg_stats[1])
        print('avg_inlier_ratio', avg_stats[2])
        print('avg_IN', avg_stats[3])
        print('avg_registration_recall', avg_stats[4])
        print('avg_FMR', avg_stats[5])
