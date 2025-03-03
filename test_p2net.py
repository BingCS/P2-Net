import os
import pickle
import cv2
import torch
import numpy as np

import time
import datetime
from os.path import exists, join

from networks.d3f import KPFCNN
from networks.d2f import D2F
from data.dl_test import P2NETMatchDataset
from data.common import P2NETCollate
from options.opt_evaluation import Options
from torch.utils.data import DataLoader
from evaluate.evaluator_p2net import Evaluator


def extract():

    # Load the dataset
    test_set = P2NETMatchDataset(opt, 'test')
    kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=P2NETCollate, **kwargs)
    print(test_set.neighborhood_limits)

    # Models
    print("******* Creating model *******")
    d2net = D2F(opt)
    d3net = KPFCNN(opt)
    d2net.to(opt.device)
    d3net.to(opt.device)

    weights = f'/home/hdd/Downloads/paotong/P2-NET/logs/2024-05-21-21-18-50_P2NET_Adam_CIRCLE_Single/models/{opt.p2net_ckpt}'
    ckpt = torch.load(weights, map_location=opt.device)
    print(f"Loaded models from {opt.p2net_ckpt}, epoch {ckpt['epoch']}")
    d2net.load_state_dict(ckpt['d2net_state_dict'])
    d3net.load_state_dict(ckpt['d3net_state_dict'])

    descriptors_path = f'/home/hdd/Downloads/paotong/P2-NET/logs/2024-05-21-21-18-50_P2NET_Adam_CIRCLE_Single/results/{opt.p2net_ckpt}_{opt.kpt_n}/descriptors'
    keypoints_path = f'/home/hdd/Downloads/paotong/P2-NET/logs/2024-05-21-21-18-50_P2NET_Adam_CIRCLE_Single/results/{opt.p2net_ckpt}_{opt.kpt_n}/keypoints'
    scores_path = f'/home/hdd/Downloads/paotong/P2-NET/logs/2024-05-21-21-18-50_P2NET_Adam_CIRCLE_Single/results/{opt.p2net_ckpt}_{opt.kpt_n}/scores'

    for test_data in test_loader:
        test_data.to(opt.device)
        with torch.no_grad():
            d2_descs, d2_kpts, d2_scores = d2net(test_data.images, test_data.valid_depth_mask, opt)
            d3_descs, d3_kpts, d3_scores = d3net(test_data, opt)

        for i, image_name in enumerate(test_data.pos_id):
            
            ptcld_name = image_name.replace('frame-', 'ptcld_')
            #ptcld_name = ptcld_name[:-3] + ptcld_name[-3:].lstrip('0')
            

            scene_seq = image_name.split("frame-")[0]
            descriptors_path_scene = join(descriptors_path, scene_seq)
            keypoints_path_scene = join(keypoints_path, scene_seq)
            scores_path_scene = join(scores_path, scene_seq)
            if not exists(descriptors_path_scene):
                os.makedirs(descriptors_path_scene)
            if not exists(keypoints_path_scene):
                os.makedirs(keypoints_path_scene)
            if not exists(scores_path_scene):
                os.makedirs(scores_path_scene)

            np.save(join(descriptors_path, f'{image_name}.color.npy'), d2_descs['pred'][i].astype(np.float32))
            np.save(join(descriptors_path, f'{ptcld_name}.npy'), d3_descs['pred'][i].astype(np.float32))
            np.save(join(keypoints_path, f'{image_name}.color.npy'), d2_kpts['pred'][i].astype(np.float32))
            np.save(join(keypoints_path, f'{ptcld_name}.npy'), d3_kpts['pred'][i].astype(np.float32))
            np.save(join(scores_path, f'{image_name}.color.npy'), d2_scores['pred'][i].astype(np.float32))
            np.save(join(scores_path, f'{ptcld_name}.npy'), d3_scores['pred'][i].astype(np.float32))

            # save random pixels
            np.save(join(descriptors_path, f'{image_name}.color_random.npy'), d2_descs['rand'][i].astype(np.float32))
            np.save(join(keypoints_path, f'{image_name}.color_random.npy'), d2_kpts['rand'][i].astype(np.float32))
            np.save(join(scores_path, f'{image_name}.color_random.npy'), d2_scores['rand'][i].astype(np.float32))

            np.save(join(descriptors_path, f'{ptcld_name}.ptcld_random.npy'), d3_descs['rand'][i].astype(np.float32))
            np.save(join(keypoints_path, f'{ptcld_name}.ptcld_random.npy'), d3_kpts['rand'][i].astype(np.float32))
            np.save(join(scores_path, f'{ptcld_name}.ptcld_random.npy'), d3_scores['rand'][i].astype(np.float32))

            print(f"Generate frame {image_name}")
            print("*" * 40)


def evaluate(results_path, data_path):
    evaluator = Evaluator(opt)
    scene_list = [
        '7-scenes-chess',
        '7-scenes-fire',
        '7-scenes-heads',
        '7-scenes-office',
        '7-scenes-pumpkin',
        '7-scenes-redkitchen',
        '7-scenes-stairs'
    ]
    kptspath = f"{results_path}/keypoints/"
    descpath = f"{results_path}/descriptors/"
    scorespath = f"{results_path}/scores/"

    scene_error = {}
    for scene in scene_list:
        scene_data_path = join(data_path, scene)
        scene_descs_path = join(descpath, scene)
        K = np.loadtxt(join(scene_data_path, 'camera-intrinsics.txt'))
        scene_error[scene] = {}
        evaluator.stats[scene] = np.array((0, 0, 0, 0, 0, 0), np.float32)

        all_seq = os.listdir(scene_descs_path)
        for seq in all_seq:
            scene_error[scene][seq] = {}
            seq_path = join(scene_descs_path, seq)
            if opt.kpts_mode == 'rand':
                imgs_desc_list = [filename for filename in os.listdir(seq_path) if filename.endswith('.color_random.npy')]
            else:
                imgs_desc_list = [filename for filename in os.listdir(seq_path) if filename.endswith('.color.npy')]

            for img_id in imgs_desc_list:
                torch.cuda.synchronize()
                t0 = time.time()
                if opt.kpts_mode == 'rand':
                    pcd_id = img_id.replace('.color_random', '.ptcld_random').replace('img', 'ptcld')
                    depth_id = img_id.replace('.color_random.npy', '.depth.png').replace('img', 'depth')
                    pose_id = img_id.replace('.color_random.npy', '.pose.txt').replace('img', 'pose')
                    print(scene, seq, img_id, pcd_id)

                    ref_descs = np.load(join(f'{descpath}/{scene}/{seq}', img_id))
                    ref_kpts = np.load(join(f'{kptspath}/{scene}/{seq}', img_id))
                    ref_scores = np.load(join(f'{scorespath}/{scene}/{seq}', img_id))
                    test_descs = np.load(join(f'{descpath}/{scene}/{seq}', pcd_id))
                    test_kpts = np.load(join(f'{kptspath}/{scene}/{seq}', pcd_id))
                    test_scores = np.load(join(f'{scorespath}/{scene}/{seq}', pcd_id))

                    ref_kpts = ref_kpts[:opt.kpt_n]
                    ref_descs = ref_descs[:opt.kpt_n]
                    test_kpts = test_kpts[:opt.kpt_n]
                    test_descs = test_descs[:opt.kpt_n]

                else:
                    # already sorted when testing
                    pcd_id = img_id.replace('.color', '').replace('frame-', 'ptcld_')

                    #print("pcd_id:", pcd_id)
                    depth_id = img_id.replace('.color.npy', '.depth.png').replace('img', 'depth')
                    #print("depth_id:", depth_id)
                    pose_id = img_id.replace('.color.npy', '.pose.txt').replace('img', 'pose')
                    print(scene, seq, img_id, pcd_id)

                    ref_descs = np.load(join(f'{descpath}/{scene}/{seq}', img_id))
                    ref_kpts = np.load(join(f'{kptspath}/{scene}/{seq}', img_id))
                    ref_scores = np.load(join(f'{scorespath}/{scene}/{seq}', img_id))
                    test_descs = np.load(join(f'{descpath}{scene}/{seq}', pcd_id))
                    test_kpts = np.load(join(f'{kptspath}/{scene}/{seq}', pcd_id))
                    test_scores = np.load(join(f'{scorespath}/{scene}/{seq}', pcd_id))

                    ref_kpts = ref_kpts[:opt.kpt_n]
                    ref_descs = ref_descs[:opt.kpt_n]
                    test_kpts = test_kpts[:opt.kpt_n]
                    test_descs = test_descs[:opt.kpt_n]

                print("raw desc shape:", ref_descs.shape, test_descs.shape)
                depth = cv2.imread(join(f'{data_path}/{scene}/{seq}', depth_id), -1)
                pose = np.loadtxt(join(f'{data_path}/{scene}/{seq}', pose_id))

                # get covisible keypoints
                test_mask = evaluator.get_covisible_mask(ref_kpts, test_kpts, depth, K, pose)
                cov_ref_coord, cov_test_coord = ref_kpts, test_kpts[test_mask]
                cov_ref_feat, cov_test_feat = ref_descs, test_descs[test_mask]
                # cov_ref_coord, cov_test_coord = ref_kpts, test_kpts
                # cov_ref_feat, cov_test_feat = ref_descs, test_descs
                num_cov_feat = min(cov_ref_coord.shape[0], cov_test_coord.shape[0])


                ################### move to GPU #########################
                depth = torch.from_numpy(depth.astype(np.int64)).to(opt.device)
                pose = torch.from_numpy(pose.astype(np.float32)).to(opt.device)
                K = torch.from_numpy(K.astype(np.float32)).to(opt.device)
                cov_ref_coord = torch.from_numpy(cov_ref_coord).to(opt.device)
                cov_test_coord = torch.from_numpy(cov_test_coord).to(opt.device)
                cov_ref_feat = torch.from_numpy(cov_ref_feat).to(opt.device)
                cov_test_feat = torch.from_numpy(cov_test_feat).to(opt.device)

                # get gt matches
                print("cov_coords shape:", cov_ref_coord.shape, cov_test_coord.shape)
                repeat, gt_num = evaluator.get_gt_matches(cov_ref_coord, cov_test_coord, depth, K, pose, opt.device)
                print("repeat:", repeat)
                print("gt_num:", gt_num)

                # establish putative matches
                if num_cov_feat > 0:
                    print("num_cov_feat:", num_cov_feat)
                    #threshold = 0.5
                    putative_matches = evaluator.feature_matcher(cov_ref_feat, cov_test_feat, opt.device)
                    print("num_putative:", len(putative_matches))
                else:
                    putative_matches = []
                num_putative = max(len(putative_matches), 1)

                # get inlier matches
                ins, irs = evaluator.get_inlier_matches(cov_ref_coord, cov_test_coord, putative_matches, depth, K, pose, opt.device)
                #inlier_matches = evaluator.get_inlier_matches(cov_ref_coord, cov_test_coord, putative_matches, depth, K, pose, opt.device)
                print("ins (list):", ins, type(ins))
                print("irs (list):", irs, type(irs))
                inlier_rate = evaluator.calculate_inlier_rate(cov_ref_coord, cov_test_coord, putative_matches, depth, K, pose, gt_num, opt.device)
                print("inlier_rate:", inlier_rate)
                num_inlier = np.sum(ins)
                print("num_inlier:", num_inlier)
                inlier_ratio = np.mean(irs)
                print ("inlier_ratio:",inlier_ratio)
                print(type(inlier_ratio))
                if inlier_ratio > opt.fmr_thld:
                    feature_matching_flag = 1.0
                else:
                    feature_matching_flag = 0.0
                print("feature_matching_flag:", feature_matching_flag)

                ################# move to CPU #################################
                cov_ref_coord = cov_ref_coord.cpu().numpy()
                cov_test_coord = cov_test_coord.cpu().numpy()
                cov_ref_feat = cov_ref_feat.cpu().numpy()
                cov_test_feat = cov_test_feat.cpu().numpy()
                putative_matches = putative_matches.cpu().numpy()
                K = K.cpu().numpy()
                pose = pose.cpu().numpy()
                depth = depth.cpu().numpy()

                # get homography accuracy
                torch.cuda.synchronize()
                t1 = time.time()
                error, correctness, mean_dist, rrs = evaluator.compute_pose_error(cov_ref_coord, cov_test_coord, putative_matches, depth, K, pose)
                scene_error[scene][seq][pcd_id] = [error, mean_dist, inlier_ratio]
                print(scene_error[scene][seq][pcd_id])
                torch.cuda.synchronize()
                t2 = time.time()
                print("pose error time:", t2 - t1)
                #rrs = evaluator.compute_pose_error(cov_ref_coord, cov_test_coord, putative_matches, depth, K, pose)
                rr = np.mean(rrs)


                evaluator.stats[scene] += np.array((1,  # counter
                                        repeat.cpu().numpy() /max(num_cov_feat, 1),  # repeatability
                                        np.mean(inlier_rate),  # inlier ratio
                                        np.mean(num_inlier),  # IN
                                        rr ,  # registration recall
                                        feature_matching_flag  # feature matching recall 
                                        ))

                torch.cuda.synchronize()
                te = time.time()
                #print("tot time:", te - t0)
                print("*" * 40)

        evaluator.stats['all_eval_stats'] += evaluator.stats[scene]
        evaluator.print_stats(scene)
        # update error after each scene
        with open(join(results_path, f'{opt.kpt_n}_stats_{opt.err_thld_point}_{opt.kpts_mode}.pkl'), 'wb') as f:
            pickle.dump(scene_error, f)

    # print all_eval_avg_stats
    evaluator.print_stats('all_eval_stats')


if __name__ == '__main__':
    opt = Options().parse()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    cuda = torch.cuda.is_available()
    opt.device = "cuda:0" if cuda else "cpu"

    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("******* Start time: {:s} ********".format(start_time))

    if opt.run == 'extractor':
        extract()
    elif opt.run == 'evaluator':
        results_path = f'/home/hdd/Downloads/paotong/P2-NET/logs/2024-05-21-21-18-50_P2NET_Adam_CIRCLE_Single/results/{opt.p2net_ckpt}_10000'
        print("kpts_mode:", opt.kpts_mode)
        print(f"results path: {results_path}")
        evaluate(results_path, opt.img_dir)
    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n******* Finished at {:s} *******".format(end_time))
