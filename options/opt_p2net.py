from options.opt_base import BaseOptions


class Options(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # common options
        self.parser.add_argument('--data', type=str, default='P2NET')
        self.parser.add_argument('--data_dir', type=str, default='./P2NET_7Scenes')
        self.parser.add_argument('--img_dir', type=str, default='./P2NET_7Scenes')
        self.parser.add_argument('--epochs', type=int, default=20)
        self.parser.add_argument('--img_mode', type=str, default='color')
        self.parser.add_argument('--img_trans_mode', type=str, default='online')
        self.parser.add_argument('--batchsize', type=int, default=1)
        self.parser.add_argument('--nThreads', type=int, default=8)
        self.parser.add_argument('--print_freq', type=int, default=1000)
        self.parser.add_argument('--save_freq', type=int, default=1)
        self.parser.add_argument('--val_size', type=int, default=500)
        self.parser.add_argument('--optm_type', type=str, default='Adam')
        self.parser.add_argument('--sgd_lr', type=float, default=1e-2)
        self.parser.add_argument('--sgd_momentum', type=float, default=0.98)
        self.parser.add_argument('--sgd_weight_decay', type=float, default=1e-6)
        self.parser.add_argument('--adam_lr', type=float, default=1e-4)
        self.parser.add_argument('--kpt_n', type=int, default=128)
        self.parser.add_argument('--out_dim', type=int, default=128)
        self.parser.add_argument('--loss_type', type=str, default='CIRCLE')
        self.parser.add_argument('--loss_mode', type=str, default='Single')
        self.parser.add_argument('--pos_margin', type=float, default=0.2)
        self.parser.add_argument('--neg_margin', type=float, default=1.4)

        # training-mode options
        self.parser.add_argument('--train_mode', type=str, default='from-scratch')
        self.parser.add_argument('--switch_epoch', type=int, default=1)
        self.parser.add_argument('--p2p_logs', type=str, default='')
        self.parser.add_argument('--d2net_logs', type=str, default='')
        self.parser.add_argument('--d3net_logs', type=str, default='')
        self.parser.add_argument('--p2p_ckpt', type=str, default='best.pth.tar')
        self.parser.add_argument('--d2net_ckpt', type=str, default='')
        self.parser.add_argument('--d3net_ckpt', type=str, default='')
        self.parser.add_argument('--d2_finetune_layers', type=list, default=['conv2', 'conv4', 'conv7_3'])
        self.parser.add_argument('--d3_finetune_layers', type=int, default=-1)

        # d2net options
        self.parser.add_argument('--safe_pixel', type=float, default=12.0)
        self.parser.add_argument('--score_thld', type=float, default=-1.0)
        self.parser.add_argument('--img_size', type=list, default=[480, 640])
        self.parser.add_argument('--comb_weights', type=list, default=[0.1667, 0.3333, 0.5])
        self.parser.add_argument('--scale', type=list, default=[3, 2, 1])
        self.parser.add_argument('--edge_thld', type=int, default=10)
        self.parser.add_argument('--nms_size', type=int, default=3)
        self.parser.add_argument('--eof_mask', type=int, default=5)

        # d3net options
        self.parser.add_argument('--safe_radius', type=float, default=0.015)
        self.parser.add_argument('--aug_rotation', type=int, default=1)
        self.parser.add_argument('--aug_noise', type=float, default=0.005)
        self.parser.add_argument('--first_subsampling_dl', type=float, default=0.015)
        self.parser.add_argument('--deform_radius', type=float, default=6.0)
        self.parser.add_argument('--conv_radius', type=float, default=2.5)
        self.parser.add_argument('--KP_extent', type=float, default=2.0)
        self.parser.add_argument('--batch_norm_momentum', type=float, default=0.98)
        self.parser.add_argument('--grad_clip_norm', type=float, default=100.0)
        self.parser.add_argument('--in_points_dim', type=int, default=3)
        self.parser.add_argument('--num_kernel_points', type=int, default=15)
        self.parser.add_argument('--first_features_dim', type=int, default=128)
        self.parser.add_argument('--in_features_dim', type=int, default=1)
        self.parser.add_argument('--fixed_kernel_points', type=str, default='center')
        self.parser.add_argument('--aggregation_mode', type=str, default='sum')
        self.parser.add_argument('--KP_influence', type=str, default='linear')
        self.parser.add_argument('--beta_mode', type=str, default='softplus')
        self.parser.add_argument('--num_layers', type=int, default=5)
        self.parser.add_argument('--architecture', type=list, default=['simple',
                                                                       'resnetb',
                                                                       'resnetb_strided',
                                                                       'resnetb',
                                                                       'resnetb',
                                                                       'resnetb_strided',
                                                                       'resnetb',
                                                                       'resnetb',
                                                                       'resnetb_strided',
                                                                       'resnetb',
                                                                       'resnetb',
                                                                       'resnetb_strided',
                                                                       'resnetb',
                                                                       'resnetb',
                                                                       'nearest_upsample',
                                                                       'unary',
                                                                       'nearest_upsample',
                                                                       'unary',
                                                                       'nearest_upsample',
                                                                       'unary',
                                                                       'nearest_upsample',
                                                                       'HeadBlock'])

        self.isTrain = True
        self.kpt_refinement = False
