# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor

try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes

import sys
sys.path.append('/workspace/changyongshu/projects/mmlab_env/for_warp_per/mmdetection3d/')
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', type=str2bool, default='False') #action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()






'''

Results writes to /tmp/tmpd9qx63a8/results/img_bbox/results_nusc.json
Evaluating bboxes of img_bbox
mAP: 0.2103                                                                                                                                                 
mATE: 0.9209
mASE: 0.2947
mAOE: 0.5716
mAVE: 1.1812
mAAE: 0.1656
NDS: 0.3099
Eval time: 229.5s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.359   0.752   0.156   0.187   1.638   0.141
truck   0.187   0.928   0.225   0.229   1.376   0.203
bus     0.215   0.982   0.223   0.231   2.403   0.311
trailer 0.063   1.146   0.290   0.943   0.695   0.091
construction_vehicle    0.040   1.129   0.471   1.064   0.198   0.258
pedestrian      0.271   0.880   0.317   0.778   0.779   0.191
motorcycle      0.193   0.889   0.301   0.675   1.648   0.123
bicycle 0.176   0.855   0.307   0.865   0.711   0.006
traffic_cone    0.309   0.829   0.357   nan     nan     nan
barrier 0.289   0.819   0.298   0.172   nan     nan
{'img_bbox_NuScenes/car_AP_dist_0.5': 0.032, 'img_bbox_NuScenes/car_AP_dist_1.0': 0.1989, 'img_bbox_NuScenes/car_AP_dist_2.0': 0.4878, 'img_bbox_NuScenes/car_AP_dist_4.0': 0.7185, 'img_bbox_NuScenes/car_trans_err': 0.7517, 'img_bbox_NuScenes/car_scale_err': 0.1563, 'img_bbox_NuScenes/car_orient_err': 0.1865, 'img_bbox_NuScenes/car_vel_err': 1.6384, 'img_bbox_NuScenes/car_attr_err': 0.1412, 'img_bbox_NuScenes/mATE': 0.9209, 'img_bbox_NuScenes/mASE': 0.2947, 'img_bbox_NuScenes/mAOE': 0.5716, 'img_bbox_NuScenes/mAVE': 1.1812, 'img_bbox_NuScenes/mAAE': 0.1656, 'img_bbox_NuScenes/truck_AP_dist_0.5': 0.0002, 'img_bbox_NuScenes/truck_AP_dist_1.0': 0.0481, 'img_bbox_NuScenes/truck_AP_dist_2.0': 0.2358, 'img_bbox_NuScenes/truck_AP_dist_4.0': 0.4651, 'img_bbox_NuScenes/truck_trans_err': 0.9284, 'img_bbox_NuScenes/truck_scale_err': 0.2248, 'img_bbox_NuScenes/truck_orient_err': 0.2286, 'img_bbox_NuScenes/truck_vel_err': 1.3759, 'img_bbox_NuScenes/truck_attr_err': 0.2027, 'img_bbox_NuScenes/trailer_AP_dist_0.5': 0.0, 'img_bbox_NuScenes/trailer_AP_dist_1.0': 0.0, 'img_bbox_NuScenes/trailer_AP_dist_2.0': 0.0303, 'img_bbox_NuScenes/trailer_AP_dist_4.0': 0.2228, 'img_bbox_NuScenes/trailer_trans_err': 1.1461, 'img_bbox_NuScenes/trailer_scale_err': 0.2902, 'img_bbox_NuScenes/trailer_orient_err': 0.9434, 'img_bbox_NuScenes/trailer_vel_err': 0.6954, 'img_bbox_NuScenes/trailer_attr_err': 0.0912, 'img_bbox_NuScenes/bus_AP_dist_0.5': 0.0, 'img_bbox_NuScenes/bus_AP_dist_1.0': 0.0403, 'img_bbox_NuScenes/bus_AP_dist_2.0': 0.2626, 'img_bbox_NuScenes/bus_AP_dist_4.0': 0.5582, 'img_bbox_NuScenes/bus_trans_err': 0.9822, 'img_bbox_NuScenes/bus_scale_err': 0.2233, 'img_bbox_NuScenes/bus_orient_err': 0.2311, 'img_bbox_NuScenes/bus_vel_err': 2.403, 'img_bbox_NuScenes/bus_attr_err': 0.3109, 'img_bbox_NuScenes/construction_vehicle_AP_dist_0.5': 0.0, 'img_bbox_NuScenes/construction_vehicle_AP_dist_1.0': 0.0, 'img_bbox_NuScenes/construction_vehicle_AP_dist_2.0': 0.0403, 'img_bbox_NuScenes/construction_vehicle_AP_dist_4.0': 0.1198, 'img_bbox_NuScenes/construction_vehicle_trans_err': 1.1288, 'img_bbox_NuScenes/construction_vehicle_scale_err': 0.4711, 'img_bbox_NuScenes/construction_vehicle_orient_err': 1.0645, 'img_bbox_NuScenes/construction_vehicle_vel_err': 0.198, 'img_bbox_NuScenes/construction_vehicle_attr_err': 0.2581, 'img_bbox_NuScenes/bicycle_AP_dist_0.5': 0.0079, 'img_bbox_NuScenes/bicycle_AP_dist_1.0': 0.0836, 'img_bbox_NuScenes/bicycle_AP_dist_2.0': 0.222, 'img_bbox_NuScenes/bicycle_AP_dist_4.0': 0.3911, 'img_bbox_NuScenes/bicycle_trans_err': 0.8548, 'img_bbox_NuScenes/bicycle_scale_err': 0.3073, 'img_bbox_NuScenes/bicycle_orient_err': 0.8654, 'img_bbox_NuScenes/bicycle_vel_err': 0.7114, 'img_bbox_NuScenes/bicycle_attr_err': 0.006, 'img_bbox_NuScenes/motorcycle_AP_dist_0.5': 0.0062, 'img_bbox_NuScenes/motorcycle_AP_dist_1.0': 0.0855, 'img_bbox_NuScenes/motorcycle_AP_dist_2.0': 0.2483, 'img_bbox_NuScenes/motorcycle_AP_dist_4.0': 0.4312, 'img_bbox_NuScenes/motorcycle_trans_err': 0.8888, 'img_bbox_NuScenes/motorcycle_scale_err': 0.3007, 'img_bbox_NuScenes/motorcycle_orient_err': 0.6747, 'img_bbox_NuScenes/motorcycle_vel_err': 1.6484, 'img_bbox_NuScenes/motorcycle_attr_err': 0.1233, 'img_bbox_NuScenes/pedestrian_AP_dist_0.5': 0.0271, 'img_bbox_NuScenes/pedestrian_AP_dist_1.0': 0.1409, 'img_bbox_NuScenes/pedestrian_AP_dist_2.0': 0.34, 'img_bbox_NuScenes/pedestrian_AP_dist_4.0': 0.578, 'img_bbox_NuScenes/pedestrian_trans_err': 0.8804, 'img_bbox_NuScenes/pedestrian_scale_err': 0.3172, 'img_bbox_NuScenes/pedestrian_orient_err': 0.7782, 'img_bbox_NuScenes/pedestrian_vel_err': 0.7791, 'img_bbox_NuScenes/pedestrian_attr_err': 0.1915, 'img_bbox_NuScenes/traffic_cone_AP_dist_0.5': 0.054, 'img_bbox_NuScenes/traffic_cone_AP_dist_1.0': 0.1852, 'img_bbox_NuScenes/traffic_cone_AP_dist_2.0': 0.3945, 'img_bbox_NuScenes/traffic_cone_AP_dist_4.0': 0.6013, 'img_bbox_NuScenes/traffic_cone_trans_err': 0.8292, 'img_bbox_NuScenes/traffic_cone_scale_err': 0.3573, 'img_bbox_NuScenes/traffic_cone_orient_err': nan, 'img_bbox_NuScenes/traffic_cone_vel_err': nan, 'img_bbox_NuScenes/traffic_cone_attr_err': nan, 'img_bbox_NuScenes/barrier_AP_dist_0.5': 0.031, 'img_bbox_NuScenes/barrier_AP_dist_1.0': 0.1879, 'img_bbox_NuScenes/barrier_AP_dist_2.0': 0.4014, 'img_bbox_NuScenes/barrier_AP_dist_4.0': 0.5358, 'img_bbox_NuScenes/barrier_trans_err': 0.8188, 'img_bbox_NuScenes/barrier_scale_err': 0.2984, 'img_bbox_NuScenes/barrier_orient_err': 0.1721, 'img_bbox_NuScenes/barrier_vel_err': nan, 'img_bbox_NuScenes/barrier_attr_err': nan, 'img_bbox_NuScenes/NDS': 0.3098847838938709, 'img_bbox_NuScenes/mAP': 0.2103298535586256}



Formating bboxes of img_bbox
Start to convert detection format...
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 36114/36114, 189.9 task/s, elapsed: 190s, ETA:     0s
Results writes to /tmp/tmp3d7isgfj/results/img_bbox/results_nusc.json
Evaluating bboxes of img_bbox
mAP: 0.2375                                                                                                                                                                                                                                                            
mATE: 0.8023
mASE: 0.2626
mAOE: 0.4961
mAVE: 1.1768
mAAE: 0.1484
NDS: 0.3478
Eval time: 113.1s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.416   0.611   0.150   0.096   1.641   0.135
truck   0.176   0.799   0.205   0.130   1.365   0.200
bus     0.239   0.989   0.194   0.147   2.482   0.277
trailer 0.062   1.155   0.231   0.806   0.548   0.082
construction_vehicle    0.041   1.154   0.443   1.128   0.144   0.216
pedestrian      0.311   0.693   0.288   0.670   0.782   0.156
motorcycle      0.209   0.745   0.266   0.557   1.685   0.121
bicycle 0.187   0.708   0.266   0.802   0.768   0.001
traffic_cone    0.455   0.563   0.318   nan     nan     nan
barrier 0.280   0.605   0.264   0.130   nan     nan
Epoch [3][2300/4434]   lr: 2.000e-03, eta: 1 day, 12:21:18, time: 3.142, data_time: 0.057, memory: 29281, loss_cls: 0.1816, loss_offset: 0.4489, loss_depth: 0.3481, loss_size: 0.4063, loss_rotsin: 0.1297, loss_centerness: 0.5671, loss_dir: 0.1915, loss_attr: 0.0933, loss: 2.3666, grad_norm: 7.9461
{'img_bbox_NuScenes/car_AP_dist_0.5': 0.0923, 'img_bbox_NuScenes/car_AP_dist_1.0': 0.2969, 'img_bbox_NuScenes/car_AP_dist_2.0': 0.5605, 'img_bbox_NuScenes/car_AP_dist_4.0': 0.714, 'img_bbox_NuScenes/car_trans_err': 0.6115, 'img_bbox_NuScenes/car_scale_err': 0.1501, 'img_bbox_NuScenes/car_orient_err': 0.0961, 'img_bbox_NuScenes/car_vel_err': 1.6407, 'img_bbox_NuScenes/car_attr_err': 0.1355, 'img_bbox_NuScenes/mATE': 0.8023, 'img_bbox_NuScenes/mASE': 0.2626, 'img_bbox_NuScenes/mAOE': 0.4961, 'img_bbox_NuScenes/mAVE': 1.1768, 'img_bbox_NuScenes/mAAE': 0.1484, 'img_bbox_NuScenes/truck_AP_dist_0.5': 0.0035, 'img_bbox_NuScenes/truck_AP_dist_1.0': 0.0766, 'img_bbox_NuScenes/truck_AP_dist_2.0': 0.2364, 'img_bbox_NuScenes/truck_AP_dist_4.0': 0.3885, 'img_bbox_NuScenes/truck_trans_err': 0.7991, 'img_bbox_NuScenes/truck_scale_err': 0.2052, 'img_bbox_NuScenes/truck_orient_err': 0.1298, 'img_bbox_NuScenes/truck_vel_err': 1.3648, 'img_bbox_NuScenes/truck_attr_err': 0.1999, 'img_bbox_NuScenes/trailer_AP_dist_0.5': 0.0, 'img_bbox_NuScenes/trailer_AP_dist_1.0': 0.0, 'img_bbox_NuScenes/trailer_AP_dist_2.0': 0.043, 'img_bbox_NuScenes/trailer_AP_dist_4.0': 0.2038, 'img_bbox_NuScenes/trailer_trans_err': 1.1552, 'img_bbox_NuScenes/trailer_scale_err': 0.2313, 'img_bbox_NuScenes/trailer_orient_err': 0.806, 'img_bbox_NuScenes/trailer_vel_err': 0.5476, 'img_bbox_NuScenes/trailer_attr_err': 0.082, 'img_bbox_NuScenes/bus_AP_dist_0.5': 0.0005, 'img_bbox_NuScenes/bus_AP_dist_1.0': 0.0651, 'img_bbox_NuScenes/bus_AP_dist_2.0': 0.3202, 'img_bbox_NuScenes/bus_AP_dist_4.0': 0.5695, 'img_bbox_NuScenes/bus_trans_err': 0.9895, 'img_bbox_NuScenes/bus_scale_err': 0.1936, 'img_bbox_NuScenes/bus_orient_err': 0.1466, 'img_bbox_NuScenes/bus_vel_err': 2.4817, 'img_bbox_NuScenes/bus_attr_err': 0.2767, 'img_bbox_NuScenes/construction_vehicle_AP_dist_0.5': 0.0, 'img_bbox_NuScenes/construction_vehicle_AP_dist_1.0': 0.0, 'img_bbox_NuScenes/construction_vehicle_AP_dist_2.0': 0.05, 'img_bbox_NuScenes/construction_vehicle_AP_dist_4.0': 0.1146, 'img_bbox_NuScenes/construction_vehicle_trans_err': 1.1539, 'img_bbox_NuScenes/construction_vehicle_scale_err': 0.4435, 'img_bbox_NuScenes/construction_vehicle_orient_err': 1.1276, 'img_bbox_NuScenes/construction_vehicle_vel_err': 0.1437, 'img_bbox_NuScenes/construction_vehicle_attr_err': 0.2162, 'img_bbox_NuScenes/bicycle_AP_dist_0.5': 0.0198, 'img_bbox_NuScenes/bicycle_AP_dist_1.0': 0.1116, 'img_bbox_NuScenes/bicycle_AP_dist_2.0': 0.2647, 'img_bbox_NuScenes/bicycle_AP_dist_4.0': 0.351, 'img_bbox_NuScenes/bicycle_trans_err': 0.7075, 'img_bbox_NuScenes/bicycle_scale_err': 0.2656, 'img_bbox_NuScenes/bicycle_orient_err': 0.8017, 'img_bbox_NuScenes/bicycle_vel_err': 0.7684, 'img_bbox_NuScenes/bicycle_attr_err': 0.0009, 'img_bbox_NuScenes/motorcycle_AP_dist_0.5': 0.0162, 'img_bbox_NuScenes/motorcycle_AP_dist_1.0': 0.1201, 'img_bbox_NuScenes/motorcycle_AP_dist_2.0': 0.2901, 'img_bbox_NuScenes/motorcycle_AP_dist_4.0': 0.4085, 'img_bbox_NuScenes/motorcycle_trans_err': 0.7446, 'img_bbox_NuScenes/motorcycle_scale_err': 0.2661, 'img_bbox_NuScenes/motorcycle_orient_err': 0.5571, 'img_bbox_NuScenes/motorcycle_vel_err': 1.6851, 'img_bbox_NuScenes/motorcycle_attr_err': 0.1205, 'img_bbox_NuScenes/pedestrian_AP_dist_0.5': 0.0519, 'img_bbox_NuScenes/pedestrian_AP_dist_1.0': 0.2009, 'img_bbox_NuScenes/pedestrian_AP_dist_2.0': 0.4152, 'img_bbox_NuScenes/pedestrian_AP_dist_4.0': 0.5744, 'img_bbox_NuScenes/pedestrian_trans_err': 0.6933, 'img_bbox_NuScenes/pedestrian_scale_err': 0.2878, 'img_bbox_NuScenes/pedestrian_orient_err': 0.67, 'img_bbox_NuScenes/pedestrian_vel_err': 0.7821, 'img_bbox_NuScenes/pedestrian_attr_err': 0.1559, 'img_bbox_NuScenes/traffic_cone_AP_dist_0.5': 0.1531, 'img_bbox_NuScenes/traffic_cone_AP_dist_1.0': 0.3948, 'img_bbox_NuScenes/traffic_cone_AP_dist_2.0': 0.5847, 'img_bbox_NuScenes/traffic_cone_AP_dist_4.0': 0.6861, 'img_bbox_NuScenes/traffic_cone_trans_err': 0.5632, 'img_bbox_NuScenes/traffic_cone_scale_err': 0.3181, 'img_bbox_NuScenes/traffic_cone_orient_err': nan, 'img_bbox_NuScenes/traffic_cone_vel_err': nan, 'img_bbox_NuScenes/traffic_cone_attr_err': nan, 'img_bbox_NuScenes/barrier_AP_dist_0.5': 0.0603, 'img_bbox_NuScenes/barrier_AP_dist_1.0': 0.2319, 'img_bbox_NuScenes/barrier_AP_dist_2.0': 0.3874, 'img_bbox_NuScenes/barrier_AP_dist_4.0': 0.4407, 'img_bbox_NuScenes/barrier_trans_err': 0.6055, 'img_bbox_NuScenes/barrier_scale_err': 0.2644, 'img_bbox_NuScenes/barrier_orient_err': 0.13, 'img_bbox_NuScenes/barrier_vel_err': nan, 'img_bbox_NuScenes/barrier_attr_err': nan, 'img_bbox_NuScenes/NDS': 0.34779682457722794, 'img_bbox_NuScenes/mAP': 0.23747999428493513}

{"mode": "train", "epoch": 4, "iter": 8850, "lr": 0.002, "memory": 8895, "data_time": 0.01434,            "loss_cls": 0.25064, "loss_offset": 0.56493, "loss_depth": 0.6805, "loss_size": 0.6631, "loss_rotsin": 0.22795, "loss_centerness": 0.57117, "loss_velo": 0.05923, "loss_dir": 0.38141, "loss_attr": 0.35015, "loss": 3.74908, "grad_norm": 11.55197, "time": 0.90603}
{"img_bbox_NuScenes/car_AP_dist_0.5": 0.0516, "img_bbox_NuScenes/car_AP_dist_1.0": 0.2345, "img_bbox_NuScenes/car_AP_dist_2.0": 0.5413, "img_bbox_NuScenes/car_AP_dist_4.0": 0.7546, "img_bbox_NuScenes/car_trans_err": 0.7103, "img_bbox_NuScenes/car_scale_err": 0.1547, "img_bbox_NuScenes/car_orient_err": 0.1651, "img_bbox_NuScenes/car_vel_err": 2.1239, "img_bbox_NuScenes/car_attr_err": 0.1494, "img_bbox_NuScenes/mATE": 0.8628, "img_bbox_NuScenes/mASE": 0.2854, "img_bbox_NuScenes/mAOE": 0.6549, "img_bbox_NuScenes/mAVE": 1.3144, "img_bbox_NuScenes/mAAE": 0.161, "img_bbox_NuScenes/truck_AP_dist_0.5": 0.0, "img_bbox_NuScenes/truck_AP_dist_1.0": 0.019, "img_bbox_NuScenes/truck_AP_dist_2.0": 0.124, "img_bbox_NuScenes/truck_AP_dist_4.0": 0.3118, "img_bbox_NuScenes/truck_trans_err": 0.936, "img_bbox_NuScenes/truck_scale_err": 0.2231, "img_bbox_NuScenes/truck_orient_err": 0.2602, "img_bbox_NuScenes/truck_vel_err": 1.4447, "img_bbox_NuScenes/truck_attr_err": 0.1952, "img_bbox_NuScenes/trailer_AP_dist_0.5": 0.0, "img_bbox_NuScenes/trailer_AP_dist_1.0": 0.0, "img_bbox_NuScenes/trailer_AP_dist_2.0": 0.0045, "img_bbox_NuScenes/trailer_AP_dist_4.0": 0.0977, "img_bbox_NuScenes/trailer_trans_err": 1.2462, "img_bbox_NuScenes/trailer_scale_err": 0.2657, "img_bbox_NuScenes/trailer_orient_err": 1.1205, "img_bbox_NuScenes/trailer_vel_err": 0.4577, "img_bbox_NuScenes/trailer_attr_err": 0.0866, "img_bbox_NuScenes/bus_AP_dist_0.5": 0.0019, "img_bbox_NuScenes/bus_AP_dist_1.0": 0.0488, "img_bbox_NuScenes/bus_AP_dist_2.0": 0.1867, "img_bbox_NuScenes/bus_AP_dist_4.0": 0.4626, "img_bbox_NuScenes/bus_trans_err": 0.8246, "img_bbox_NuScenes/bus_scale_err": 0.2129, "img_bbox_NuScenes/bus_orient_err": 0.2632, "img_bbox_NuScenes/bus_vel_err": 2.4379, "img_bbox_NuScenes/bus_attr_err": 0.3431, "img_bbox_NuScenes/construction_vehicle_AP_dist_0.5": 0.0, "img_bbox_NuScenes/construction_vehicle_AP_dist_1.0": 0.0, "img_bbox_NuScenes/construction_vehicle_AP_dist_2.0": 0.0112, "img_bbox_NuScenes/construction_vehicle_AP_dist_4.0": 0.0838, "img_bbox_NuScenes/construction_vehicle_trans_err": 1.0872, "img_bbox_NuScenes/construction_vehicle_scale_err": 0.5303, "img_bbox_NuScenes/construction_vehicle_orient_err": 1.2029, "img_bbox_NuScenes/construction_vehicle_vel_err": 0.1182, "img_bbox_NuScenes/construction_vehicle_attr_err": 0.2353, "img_bbox_NuScenes/bicycle_AP_dist_0.5": 0.0113, "img_bbox_NuScenes/bicycle_AP_dist_1.0": 0.0901, "img_bbox_NuScenes/bicycle_AP_dist_2.0": 0.2449, "img_bbox_NuScenes/bicycle_AP_dist_4.0": 0.3791, "img_bbox_NuScenes/bicycle_trans_err": 0.8209, "img_bbox_NuScenes/bicycle_scale_err": 0.2792, "img_bbox_NuScenes/bicycle_orient_err": 0.993, "img_bbox_NuScenes/bicycle_vel_err": 0.8963, "img_bbox_NuScenes/bicycle_attr_err": 0.0239, "img_bbox_NuScenes/motorcycle_AP_dist_0.5": 0.0111, "img_bbox_NuScenes/motorcycle_AP_dist_1.0": 0.111, "img_bbox_NuScenes/motorcycle_AP_dist_2.0": 0.2857, "img_bbox_NuScenes/motorcycle_AP_dist_4.0": 0.4439, "img_bbox_NuScenes/motorcycle_trans_err": 0.8255, "img_bbox_NuScenes/motorcycle_scale_err": 0.2707, "img_bbox_NuScenes/motorcycle_orient_err": 0.7562, "img_bbox_NuScenes/motorcycle_vel_err": 2.1074, "img_bbox_NuScenes/motorcycle_attr_err": 0.0831, "img_bbox_NuScenes/pedestrian_AP_dist_0.5": 0.0598, "img_bbox_NuScenes/pedestrian_AP_dist_1.0": 0.2282, "img_bbox_NuScenes/pedestrian_AP_dist_2.0": 0.4553, "img_bbox_NuScenes/pedestrian_AP_dist_4.0": 0.6403, "img_bbox_NuScenes/pedestrian_trans_err": 0.7634, "img_bbox_NuScenes/pedestrian_scale_err": 0.2942, "img_bbox_NuScenes/pedestrian_orient_err": 0.9239, "img_bbox_NuScenes/pedestrian_vel_err": 0.9289, "img_bbox_NuScenes/pedestrian_attr_err": 0.1716, "img_bbox_NuScenes/traffic_cone_AP_dist_0.5": 0.1306, "img_bbox_NuScenes/traffic_cone_AP_dist_1.0": 0.3558, "img_bbox_NuScenes/traffic_cone_AP_dist_2.0": 0.5635, "img_bbox_NuScenes/traffic_cone_AP_dist_4.0": 0.6848, "img_bbox_NuScenes/traffic_cone_trans_err": 0.6586, "img_bbox_NuScenes/traffic_cone_scale_err": 0.333, "img_bbox_NuScenes/traffic_cone_orient_err": NaN, "img_bbox_NuScenes/traffic_cone_vel_err": NaN, "img_bbox_NuScenes/traffic_cone_attr_err": NaN, "img_bbox_NuScenes/barrier_AP_dist_0.5": 0.0504, "img_bbox_NuScenes/barrier_AP_dist_1.0": 0.2491, "img_bbox_NuScenes/barrier_AP_dist_2.0": 0.51, "img_bbox_NuScenes/barrier_AP_dist_4.0": 0.6193, "img_bbox_NuScenes/barrier_trans_err": 0.755, "img_bbox_NuScenes/barrier_scale_err": 0.2899, "img_bbox_NuScenes/barrier_orient_err": 0.2091, "img_bbox_NuScenes/barrier_vel_err": NaN, "img_bbox_NuScenes/barrier_attr_err": NaN, "img_bbox_NuScenes/NDS": 0.31682, "img_bbox_NuScenes/mAP": 0.22646}     "mode": "val", "epoch": 4, "iter": 4515, "lr": 0.002, 
{"img_bbox_NuScenes/car_AP_dist_0.5": 0.0946, "img_bbox_NuScenes/car_AP_dist_1.0": 0.3345, "img_bbox_NuScenes/car_AP_dist_2.0": 0.6244, "img_bbox_NuScenes/car_AP_dist_4.0": 0.7996, "img_bbox_NuScenes/car_trans_err": 0.6246, "img_bbox_NuScenes/car_scale_err": 0.15, "img_bbox_NuScenes/car_orient_err": 0.1099, "img_bbox_NuScenes/car_vel_err": 2.0157, "img_bbox_NuScenes/car_attr_err": 0.1324, "img_bbox_NuScenes/mATE": 0.7901, "img_bbox_NuScenes/mASE": 0.2606, "img_bbox_NuScenes/mAOE": 0.4994, "img_bbox_NuScenes/mAVE": 1.2859, "img_bbox_NuScenes/mAAE": 0.1666, "img_bbox_NuScenes/truck_AP_dist_0.5": 0.0031, "img_bbox_NuScenes/truck_AP_dist_1.0": 0.0744, "img_bbox_NuScenes/truck_AP_dist_2.0": 0.28, "img_bbox_NuScenes/truck_AP_dist_4.0": 0.5076, "img_bbox_NuScenes/truck_trans_err": 0.8676, "img_bbox_NuScenes/truck_scale_err": 0.1969, "img_bbox_NuScenes/truck_orient_err": 0.1416, "img_bbox_NuScenes/truck_vel_err": 1.389, "img_bbox_NuScenes/truck_attr_err": 0.1797, "img_bbox_NuScenes/trailer_AP_dist_0.5": 0.0, "img_bbox_NuScenes/trailer_AP_dist_1.0": 0.0002, "img_bbox_NuScenes/trailer_AP_dist_2.0": 0.0838, "img_bbox_NuScenes/trailer_AP_dist_4.0": 0.3087, "img_bbox_NuScenes/trailer_trans_err": 1.1499, "img_bbox_NuScenes/trailer_scale_err": 0.2289, "img_bbox_NuScenes/trailer_orient_err": 0.7075, "img_bbox_NuScenes/trailer_vel_err": 0.4647, "img_bbox_NuScenes/trailer_attr_err": 0.1113, "img_bbox_NuScenes/bus_AP_dist_0.5": 0.0061, "img_bbox_NuScenes/bus_AP_dist_1.0": 0.0979, "img_bbox_NuScenes/bus_AP_dist_2.0": 0.3794, "img_bbox_NuScenes/bus_AP_dist_4.0": 0.6468, "img_bbox_NuScenes/bus_trans_err": 0.874, "img_bbox_NuScenes/bus_scale_err": 0.1851, "img_bbox_NuScenes/bus_orient_err": 0.225, "img_bbox_NuScenes/bus_vel_err": 2.6745, "img_bbox_NuScenes/bus_attr_err": 0.33, "img_bbox_NuScenes/construction_vehicle_AP_dist_0.5": 0.0, "img_bbox_NuScenes/construction_vehicle_AP_dist_1.0": 0.0008, "img_bbox_NuScenes/construction_vehicle_AP_dist_2.0": 0.0431, "img_bbox_NuScenes/construction_vehicle_AP_dist_4.0": 0.1562, "img_bbox_NuScenes/construction_vehicle_trans_err": 0.9664, "img_bbox_NuScenes/construction_vehicle_scale_err": 0.4401, "img_bbox_NuScenes/construction_vehicle_orient_err": 1.106, "img_bbox_NuScenes/construction_vehicle_vel_err": 0.1208, "img_bbox_NuScenes/construction_vehicle_attr_err": 0.3401, "img_bbox_NuScenes/bicycle_AP_dist_0.5": 0.0421, "img_bbox_NuScenes/bicycle_AP_dist_1.0": 0.1741, "img_bbox_NuScenes/bicycle_AP_dist_2.0": 0.3644, "img_bbox_NuScenes/bicycle_AP_dist_4.0": 0.5009, "img_bbox_NuScenes/bicycle_trans_err": 0.7308, "img_bbox_NuScenes/bicycle_scale_err": 0.2693, "img_bbox_NuScenes/bicycle_orient_err": 0.7499, "img_bbox_NuScenes/bicycle_vel_err": 0.855, "img_bbox_NuScenes/bicycle_attr_err": 0.0116, "img_bbox_NuScenes/motorcycle_AP_dist_0.5": 0.0329, "img_bbox_NuScenes/motorcycle_AP_dist_1.0": 0.1604, "img_bbox_NuScenes/motorcycle_AP_dist_2.0": 0.3758, "img_bbox_NuScenes/motorcycle_AP_dist_4.0": 0.5181, "img_bbox_NuScenes/motorcycle_trans_err": 0.7846, "img_bbox_NuScenes/motorcycle_scale_err": 0.2556, "img_bbox_NuScenes/motorcycle_orient_err": 0.5872, "img_bbox_NuScenes/motorcycle_vel_err": 1.871, "img_bbox_NuScenes/motorcycle_attr_err": 0.0738, "img_bbox_NuScenes/pedestrian_AP_dist_0.5": 0.0849, "img_bbox_NuScenes/pedestrian_AP_dist_1.0": 0.2851, "img_bbox_NuScenes/pedestrian_AP_dist_2.0": 0.529, "img_bbox_NuScenes/pedestrian_AP_dist_4.0": 0.6948, "img_bbox_NuScenes/pedestrian_trans_err": 0.7106, "img_bbox_NuScenes/pedestrian_scale_err": 0.2875, "img_bbox_NuScenes/pedestrian_orient_err": 0.7283, "img_bbox_NuScenes/pedestrian_vel_err": 0.8962, "img_bbox_NuScenes/pedestrian_attr_err": 0.1541, "img_bbox_NuScenes/traffic_cone_AP_dist_0.5": 0.2026, "img_bbox_NuScenes/traffic_cone_AP_dist_1.0": 0.47, "img_bbox_NuScenes/traffic_cone_AP_dist_2.0": 0.6458, "img_bbox_NuScenes/traffic_cone_AP_dist_4.0": 0.7339, "img_bbox_NuScenes/traffic_cone_trans_err": 0.5499, "img_bbox_NuScenes/traffic_cone_scale_err": 0.3196, "img_bbox_NuScenes/traffic_cone_orient_err": NaN, "img_bbox_NuScenes/traffic_cone_vel_err": NaN, "img_bbox_NuScenes/traffic_cone_attr_err": NaN, "img_bbox_NuScenes/barrier_AP_dist_0.5": 0.1039, "img_bbox_NuScenes/barrier_AP_dist_1.0": 0.3398, "img_bbox_NuScenes/barrier_AP_dist_2.0": 0.5642, "img_bbox_NuScenes/barrier_AP_dist_4.0": 0.6525, "img_bbox_NuScenes/barrier_trans_err": 0.6421, "img_bbox_NuScenes/barrier_scale_err": 0.2729, "img_bbox_NuScenes/barrier_orient_err": 0.1395, "img_bbox_NuScenes/barrier_vel_err": NaN, "img_bbox_NuScenes/barrier_attr_err": NaN, "img_bbox_NuScenes/NDS": 0.37729, "img_bbox_NuScenes/mAP": 0.29792}  "mode": "val", "epoch": 12, "iter": 4515, "lr": 0.002, 
mAP: 0.29792                                                                                                                                                                                                                                                           
mATE: 0.7901
mASE: 0.2606
mAOE: 0.4994
mAVE: 1.2859
mAAE: 0.1666
NDS: 0.3772





model.train()下测试的
Evaluating bboxes of img_bbox
mAP: 0.1660                                                                                                                                                                                                                                                                                                                                       
mATE: 0.9057
mASE: 0.2891
mAOE: 0.5376
mAVE: 1.2232
mAAE: 0.1637
NDS: 0.2934
Eval time: 128.1s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.320   0.732   0.154   0.176   1.669   0.142
truck   0.141   0.916   0.224   0.197   1.426   0.205
bus     0.179   0.978   0.221   0.219   2.374   0.306
trailer 0.024   1.128   0.284   0.902   0.762   0.114
construction_vehicle    0.021   1.144   0.453   1.036   0.185   0.232
pedestrian      0.226   0.852   0.313   0.731   0.790   0.176
motorcycle      0.137   0.868   0.289   0.601   1.794   0.135
bicycle 0.132   0.825   0.304   0.818   0.785   0.001
traffic_cone    0.280   0.815   0.354   nan     nan     nan
barrier 0.200   0.799   0.295   0.159   nan     nan
{'img_bbox_NuScenes/car_AP_dist_0.5': 0.031, 'img_bbox_NuScenes/car_AP_dist_1.0': 0.1811, 'img_bbox_NuScenes/car_AP_dist_2.0': 0.4308, 'img_bbox_NuScenes/car_AP_dist_4.0': 0.6374, 'img_bbox_NuScenes/car_trans_err': 0.7316, 'img_bbox_NuScenes/car_scale_err': 0.1544, 'img_bbox_NuScenes/car_orient_err': 0.1758, 'img_bbox_NuScenes/car_vel_err': 1.6695, 'img_bbox_NuScenes/car_attr_err': 0.1416, 'img_bbox_NuScenes/mATE': 0.9057, 'img_bbox_NuScenes/mASE': 0.2891, 'img_bbox_NuScenes/mAOE': 0.5376, 'img_bbox_NuScenes/mAVE': 1.2232, 'img_bbox_NuScenes/mAAE': 0.1637, 'img_bbox_NuScenes/truck_AP_dist_0.5': 0.0, 'img_bbox_NuScenes/truck_AP_dist_1.0': 0.0362, 'img_bbox_NuScenes/truck_AP_dist_2.0': 0.18, 'img_bbox_NuScenes/truck_AP_dist_4.0': 0.349, 'img_bbox_NuScenes/truck_trans_err': 0.9165, 'img_bbox_NuScenes/truck_scale_err': 0.2243, 'img_bbox_NuScenes/truck_orient_err': 0.1966, 'img_bbox_NuScenes/truck_vel_err': 1.4258, 'img_bbox_NuScenes/truck_attr_err': 0.2047, 'img_bbox_NuScenes/trailer_AP_dist_0.5': 0.0, 'img_bbox_NuScenes/trailer_AP_dist_1.0': 0.0, 'img_bbox_NuScenes/trailer_AP_dist_2.0': 0.0055, 'img_bbox_NuScenes/trailer_AP_dist_4.0': 0.0901, 'img_bbox_NuScenes/trailer_trans_err': 1.1276, 'img_bbox_NuScenes/trailer_scale_err': 0.2838, 'img_bbox_NuScenes/trailer_orient_err': 0.9025, 'img_bbox_NuScenes/trailer_vel_err': 0.7623, 'img_bbox_NuScenes/trailer_attr_err': 0.1137, 'img_bbox_NuScenes/bus_AP_dist_0.5': 0.0, 'img_bbox_NuScenes/bus_AP_dist_1.0': 0.033, 'img_bbox_NuScenes/bus_AP_dist_2.0': 0.2223, 'img_bbox_NuScenes/bus_AP_dist_4.0': 0.4603, 'img_bbox_NuScenes/bus_trans_err': 0.9781, 'img_bbox_NuScenes/bus_scale_err': 0.2211, 'img_bbox_NuScenes/bus_orient_err': 0.219, 'img_bbox_NuScenes/bus_vel_err': 2.3739, 'img_bbox_NuScenes/bus_attr_err': 0.3058, 'img_bbox_NuScenes/construction_vehicle_AP_dist_0.5': 0.0, 'img_bbox_NuScenes/construction_vehicle_AP_dist_1.0': 0.0, 'img_bbox_NuScenes/construction_vehicle_AP_dist_2.0': 0.0179, 'img_bbox_NuScenes/construction_vehicle_AP_dist_4.0': 0.0661, 'img_bbox_NuScenes/construction_vehicle_trans_err': 1.1439, 'img_bbox_NuScenes/construction_vehicle_scale_err': 0.4528, 'img_bbox_NuScenes/construction_vehicle_orient_err': 1.0361, 'img_bbox_NuScenes/construction_vehicle_vel_err': 0.1855, 'img_bbox_NuScenes/construction_vehicle_attr_err': 0.2316, 'img_bbox_NuScenes/bicycle_AP_dist_0.5': 0.0042, 'img_bbox_NuScenes/bicycle_AP_dist_1.0': 0.0549, 'img_bbox_NuScenes/bicycle_AP_dist_2.0': 0.1598, 'img_bbox_NuScenes/bicycle_AP_dist_4.0': 0.3088, 'img_bbox_NuScenes/bicycle_trans_err': 0.8251, 'img_bbox_NuScenes/bicycle_scale_err': 0.304, 'img_bbox_NuScenes/bicycle_orient_err': 0.8176, 'img_bbox_NuScenes/bicycle_vel_err': 0.7845, 'img_bbox_NuScenes/bicycle_attr_err': 0.0006, 'img_bbox_NuScenes/motorcycle_AP_dist_0.5': 0.0029, 'img_bbox_NuScenes/motorcycle_AP_dist_1.0': 0.0522, 'img_bbox_NuScenes/motorcycle_AP_dist_2.0': 0.1678, 'img_bbox_NuScenes/motorcycle_AP_dist_4.0': 0.3256, 'img_bbox_NuScenes/motorcycle_trans_err': 0.8678, 'img_bbox_NuScenes/motorcycle_scale_err': 0.2889, 'img_bbox_NuScenes/motorcycle_orient_err': 0.6009, 'img_bbox_NuScenes/motorcycle_vel_err': 1.7938, 'img_bbox_NuScenes/motorcycle_attr_err': 0.1352, 'img_bbox_NuScenes/pedestrian_AP_dist_0.5': 0.0184, 'img_bbox_NuScenes/pedestrian_AP_dist_1.0': 0.1021, 'img_bbox_NuScenes/pedestrian_AP_dist_2.0': 0.2778, 'img_bbox_NuScenes/pedestrian_AP_dist_4.0': 0.5072, 'img_bbox_NuScenes/pedestrian_trans_err': 0.8522, 'img_bbox_NuScenes/pedestrian_scale_err': 0.3129, 'img_bbox_NuScenes/pedestrian_orient_err': 0.7309, 'img_bbox_NuScenes/pedestrian_vel_err': 0.79, 'img_bbox_NuScenes/pedestrian_attr_err': 0.1765, 'img_bbox_NuScenes/traffic_cone_AP_dist_0.5': 0.043, 'img_bbox_NuScenes/traffic_cone_AP_dist_1.0': 0.1572, 'img_bbox_NuScenes/traffic_cone_AP_dist_2.0': 0.355, 'img_bbox_NuScenes/traffic_cone_AP_dist_4.0': 0.5655, 'img_bbox_NuScenes/traffic_cone_trans_err': 0.8152, 'img_bbox_NuScenes/traffic_cone_scale_err': 0.3538, 'img_bbox_NuScenes/traffic_cone_orient_err': nan, 'img_bbox_NuScenes/traffic_cone_vel_err': nan, 'img_bbox_NuScenes/traffic_cone_attr_err': nan, 'img_bbox_NuScenes/barrier_AP_dist_0.5': 0.017, 'img_bbox_NuScenes/barrier_AP_dist_1.0': 0.1176, 'img_bbox_NuScenes/barrier_AP_dist_2.0': 0.2765, 'img_bbox_NuScenes/barrier_AP_dist_4.0': 0.3874, 'img_bbox_NuScenes/barrier_trans_err': 0.7986, 'img_bbox_NuScenes/barrier_scale_err': 0.2948, 'img_bbox_NuScenes/barrier_orient_err': 0.1588, 'img_bbox_NuScenes/barrier_vel_err': nan, 'img_bbox_NuScenes/barrier_attr_err': nan, 'img_bbox_NuScenes/NDS': 0.2934200859019954, 'img_bbox_NuScenes/mAP': 0.166045877723416}

'''

