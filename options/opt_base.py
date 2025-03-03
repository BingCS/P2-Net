import os
import shutil
import argparse
import datetime
from utils import base_tools


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):
        # base options
        self.parser.add_argument('--gpus', type=str, default='0')
        self.parser.add_argument('--logdir', type=str, default='./logs')
        self.parser.add_argument('--exp_name', type=str, default=None)
        self.parser.add_argument('--models_dir', type=str, default='models')
        self.parser.add_argument('--runs_dir', type=str, default='runs')
        self.parser.add_argument('--results_dir', type=str, default='results')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain
        self.opt.kpt_refinement = self.kpt_refinement

        if self.opt.data == '3DMatch' or 'P2P':
            self.opt.use_batch_norm = False
            self.opt.modulated = False

        # save to the disk
        if self.opt.isTrain:
            self.opt.kpt_refinement = False
            if self.opt.exp_name is None:
                self.opt.exp_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_' + self.opt.data + '_' + self.opt.optm_type + '_' + self.opt.loss_type + '_' + self.opt.loss_mode
            expr_dir = os.path.join(self.opt.logdir, self.opt.exp_name)
            self.opt.models_dir = os.path.join(expr_dir, self.opt.models_dir)
            self.opt.runs_dir = os.path.join(expr_dir, self.opt.runs_dir)
            self.opt.results_dir = os.path.join(expr_dir, self.opt.results_dir)
            base_tools.mkdirs([self.opt.logdir, expr_dir, self.opt.runs_dir, self.opt.models_dir, self.opt.results_dir])

            # save training scripts
            if self.opt.data == 'P2P':
                shutil.copy2(os.path.join('.', 'train_p2p.py'), os.path.join(expr_dir, 'train.py'))
                shutil.copy2(os.path.join('networks', 'd2f.py'), os.path.join(expr_dir, 'd2f.py'))
                shutil.copy2(os.path.join('networks', 'd3f.py'), os.path.join(expr_dir, 'd3f.py'))
                shutil.copy2(os.path.join('networks/libs', 'd2_lib.py'), os.path.join(expr_dir, 'd2_lib.py'))
                shutil.copy2(os.path.join('networks/libs', 'kpconv_lib.py'), os.path.join(expr_dir, 'kpconv_lib.py'))
                shutil.copy2(os.path.join('networks', 'loss_p2p.py'), os.path.join(expr_dir, 'loss_p2p.py'))
                shutil.copy2(os.path.join('options', 'opt_p2p.py'), os.path.join(expr_dir, 'opt_p2p.py'))
                shutil.copy2(os.path.join('data', 'common.py'), os.path.join(expr_dir, 'common.py'))
                shutil.copy2(os.path.join('data', 'dl_p2p.py'), os.path.join(expr_dir, 'dl_p2p.py'))

            elif self.opt.data == '3DMatch':
                shutil.copy2(os.path.join('.', 'train_3dmatch.py'), os.path.join(expr_dir, 'train.py'))
                shutil.copy2(os.path.join('networks', 'kpconv.py'), os.path.join(expr_dir, 'kpconv.py'))
                shutil.copy2(os.path.join('networks/libs', 'kpconv_lib.py'), os.path.join(expr_dir, 'kpconv_lib.py'))
                shutil.copy2(os.path.join('networks', 'loss_3dmatch.py'), os.path.join(expr_dir, 'loss_3dmatch.py'))
                shutil.copy2(os.path.join('options', 'opt_3dmatch.py'), os.path.join(expr_dir, 'opt_3dmatch.py'))
                shutil.copy2(os.path.join('data', 'common.py'), os.path.join(expr_dir, 'common.py'))
                shutil.copy2(os.path.join('data', 'dl_3dmatch.py'), os.path.join(expr_dir, 'dl_3dmatch.py'))

            elif self.opt.data == 'GL3D':
                shutil.copy2(os.path.join('.', 'train_gl3d.py'), os.path.join(expr_dir, 'train.py'))
                shutil.copy2(os.path.join('networks', 'd2f.py'), os.path.join(expr_dir, 'd2f.py'))
                shutil.copy2(os.path.join('networks/libs', 'd2_lib.py'), os.path.join(expr_dir, 'd2_lib.py'))
                shutil.copy2(os.path.join('networks', 'loss_gl3d.py'), os.path.join(expr_dir, 'loss_gl3d.py'))
                shutil.copy2(os.path.join('options', 'opt_gl3d.py'), os.path.join(expr_dir, 'opt_gl3d.py'))
                shutil.copy2(os.path.join('data', 'dl_gl3d.py'), os.path.join(expr_dir, 'dl_gl3d.py'))

            args = vars(self.opt)
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ---------------')

        return self.opt
