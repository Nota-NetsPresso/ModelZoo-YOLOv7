import argparse
import os
from pathlib import Path
import random
import time
from copy import deepcopy

import numpy as np
import yaml
from loguru import logger
import torch
import torch.fx as fx
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import onnx

from netspresso.compressor import ModelCompressor, Task, Framework

from utils.torch_utils import intersect_dicts
from utils.general import increment_path, fitness, get_latest_run, check_file, \
    print_mutation, set_logging, colorstr, check_img_size
from utils.plots import plot_evolution
from utils.torch_utils import select_device, intersect_dicts, is_parallel
from utils.activations import Hardswish, SiLU
from utils.wandb_logging.wandb_utils import check_wandb_resume
from utils.add_nms import RegisterNMS
import models
from models.yolo import Model
from models.common import *
from models.experimental import attempt_load, End2End
from train import train
from train_aux import train as train_aux
from yolov7_fx2p import fx2p


def train_run(opt):
    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    #if opt.global_rank in [-1, 0]:
    #    check_git_status()
    #    check_requirements()

    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        if opt.name in ['yolov7', 'yolov7x']: # Use different func
            train(hyp.copy(), opt, device, tb_writer)
        else:
            train_aux(hyp.copy(), opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),   # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0),  # segment copy-paste (probability)
                'paste_in': (1, 0.0, 1.0)}    # segment copy-paste (probability)
        
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
                
        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            if opt.name in ['yolov7', 'yolov7x']: # Use different func
                results = train(hyp.copy(), opt, device)
            else:
                results = train_aux(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
    
    return opt

def export_onnx(opt, save_path):
    device = select_device(opt.export_device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(1, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = not opt.grid  # set Detect() layer grid export
    y = model(img)  # dry run
    if opt.include_nms:
        model.model[-1].include_nms = True
        y = None

    print(model) # check
    model.eval()
    output_names = ['classes', 'boxes'] if y is None else ['output']
    dynamic_axes = None
    if opt.dynamic:
        dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
            'output': {0: 'batch', 2: 'y', 3: 'x'}}
    if opt.dynamic_batch:
        opt.batch_size = 'batch'
        dynamic_axes = {
            'images': {
                0: 'batch',
            }, }
        if opt.end2end and opt.max_wh is None:
            output_axes = {
                'num_dets': {0: 'batch'},
                'det_boxes': {0: 'batch'},
                'det_scores': {0: 'batch'},
                'det_classes': {0: 'batch'},
            }
        else:
            output_axes = {
                'output': {0: 'batch'},
            }
        dynamic_axes.update(output_axes)
    if opt.grid:
        if opt.end2end:
            print('\nStarting export end2end onnx model for %s...' % 'TensorRT' if opt.max_wh is None else 'onnxruntime')
            model = End2End(model,opt.topk_all,opt.iou_thres,opt.conf_thres,opt.max_wh,device,len(labels))
            if opt.end2end and opt.max_wh is None:
                output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                shapes = [opt.batch_size, 1, opt.batch_size, opt.topk_all, 4,
                            opt.batch_size, opt.topk_all, opt.batch_size, opt.topk_all]
            else:
                output_names = ['output']
        else:
            model.model[-1].concat = True

    torch.onnx.export(model, img, save_path, verbose=False, opset_version=12, input_names=['images'],
                        output_names=output_names,
                        dynamic_axes=dynamic_axes)

    # Checks
    onnx_model = onnx.load(save_path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    if opt.end2end and opt.max_wh is None:
        for i in onnx_model.graph.output:
            for j in i.type.tensor_type.shape.dim:
                j.dim_param = str(shapes.pop(0))

    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model

    # # Metadata
    # d = {'stride': int(max(model.stride))}
    # for k, v in d.items():
    #     meta = onnx_model.metadata_props.add()
    #     meta.key, meta.value = k, str(v)
    # onnx.save(onnx_model, f)

    if opt.simplify:
        try:
            import onnxsim

            print('\nStarting to simplify ONNX...')
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplifier failure: {e}')

    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    onnx.save(onnx_model, save_path)
    print('ONNX export success, saved as %s' % save_path)

    if opt.include_nms:
        print('Registering NMS plugin for ONNX...')
        mo = RegisterNMS(save_path)
        mo.register_nms()
        mo.save(save_path)

def reparam(opt):
    device = select_device('cpu', batch_size=1)
    # model trained by cfg/training/*.yaml
    ckpt = torch.load(opt.weights, map_location=device)
    # reparameterized model in cfg/deploy/*.yaml
    model = Model('cfg/deploy/' + opt.name + '.yaml', ch=3, nc=80).to(device)

    with open('cfg/deploy/' + opt.name + '.yaml') as f:
        yml = yaml.load(f, Loader=yaml.SafeLoader)
    anchors = len(yml['anchors'][0]) // 2

    state_dict = ckpt['model'].float().state_dict()

    if opt.name == 'yolov7':
        # check NetsPresso FD
        for i in state_dict:
            assert (f'model.105.m' in i and 'netspressofds' in i) == False, 'Reparameterization is not possible after using NetsPresso FD because the model structure has changed'

        # copy intersect weights
        exclude = []
        check_layer = []

        for k, v in state_dict.items():
            if k in model.state_dict() and not any(x in k for x in exclude):
                name_k = k.split('.')
                
                if ''.join(name_k[1:-1]) not in check_layer:
                    check_layer.append(''.join(name_k[1:-1]))
                    
                    if len(name_k) == 4:
                        if isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]], torch.nn.modules.conv.Conv2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]].in_channels = v.size(1)
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]].out_channels = v.size(0)
                        elif isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]], torch.nn.modules.batchnorm.BatchNorm2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]].num_features = v.size(0)
                    
                    elif len(name_k) == 5:
                        if isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]], torch.nn.modules.conv.Conv2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]].in_channels = v.size(1)
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]].out_channels = v.size(0)
                        elif isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]], torch.nn.modules.batchnorm.BatchNorm2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]].num_features = v.size(0)

        for k, v in state_dict.items():
            if k in model.state_dict() and not any(x in k for x in exclude):
                name_k = k.split('.')
                if v.dtype == torch.int64:
                    apply_v = torch.tensor(v)
                else:
                    apply_v = nn.Parameter(v)
                
                if len(name_k) == 4:
                    setattr(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]], name_k[3], apply_v) 
                elif len(name_k) == 5:
                    setattr(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]], name_k[4], apply_v)

        model.names = ckpt['model'].names
        model.nc = ckpt['model'].nc

        # reparametrized YOLOR
        for i in range((model.nc+5)*anchors):
            model.state_dict()['model.105.m.0.weight'].data[i, :, :, :] *= state_dict['model.105.im.0.implicit'].data[:, i, : :].squeeze()
            model.state_dict()['model.105.m.1.weight'].data[i, :, :, :] *= state_dict['model.105.im.1.implicit'].data[:, i, : :].squeeze()
            model.state_dict()['model.105.m.2.weight'].data[i, :, :, :] *= state_dict['model.105.im.2.implicit'].data[:, i, : :].squeeze()
        model.state_dict()['model.105.m.0.bias'].data += state_dict['model.105.m.0.weight'].mul(state_dict['model.105.ia.0.implicit']).sum(1).squeeze()
        model.state_dict()['model.105.m.1.bias'].data += state_dict['model.105.m.1.weight'].mul(state_dict['model.105.ia.1.implicit']).sum(1).squeeze()
        model.state_dict()['model.105.m.2.bias'].data += state_dict['model.105.m.2.weight'].mul(state_dict['model.105.ia.2.implicit']).sum(1).squeeze()
        model.state_dict()['model.105.m.0.bias'].data *= state_dict['model.105.im.0.implicit'].data.squeeze()
        model.state_dict()['model.105.m.1.bias'].data *= state_dict['model.105.im.1.implicit'].data.squeeze()
        model.state_dict()['model.105.m.2.bias'].data *= state_dict['model.105.im.2.implicit'].data.squeeze()

    elif opt.name == 'yolov7x':
        # check NetsPresso FD
        for i in state_dict:
            assert (f'model.121.m' in i and 'netspressofds' in i) == False, 'Reparameterization is not possible after using NetsPresso FD because the model structure has changed'
            
        # copy intersect weights   
        exclude = []
        check_layer = []

        for k, v in state_dict.items():
            if k in model.state_dict() and not any(x in k for x in exclude):
                name_k = k.split('.')
                
                if ''.join(name_k[1:-1]) not in check_layer:
                    check_layer.append(''.join(name_k[1:-1]))
                    
                    if len(name_k) == 4:
                        if isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]], torch.nn.modules.conv.Conv2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]].in_channels = v.size(1)
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]].out_channels = v.size(0)
                        elif isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]], torch.nn.modules.batchnorm.BatchNorm2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]].num_features = v.size(0)
                    
                    elif len(name_k) == 5:
                        if isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]], torch.nn.modules.conv.Conv2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]].in_channels = v.size(1)
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]].out_channels = v.size(0)
                        elif isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]], torch.nn.modules.batchnorm.BatchNorm2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]].num_features = v.size(0)

        for k, v in state_dict.items():
            if k in model.state_dict() and not any(x in k for x in exclude):
                name_k = k.split('.')
                if v.dtype == torch.int64:
                    apply_v = torch.tensor(v)
                else:
                    apply_v = nn.Parameter(v)
                
                if len(name_k) == 4:
                    setattr(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]], name_k[3], apply_v) 
                elif len(name_k) == 5:
                    setattr(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]], name_k[4], apply_v)

        model.names = ckpt['model'].names
        model.nc = ckpt['model'].nc

        # reparametrized YOLOR
        for i in range((model.nc+5)*anchors):
            model.state_dict()['model.121.m.0.weight'].data[i, :, :, :] *= state_dict['model.121.im.0.implicit'].data[:, i, : :].squeeze()
            model.state_dict()['model.121.m.1.weight'].data[i, :, :, :] *= state_dict['model.121.im.1.implicit'].data[:, i, : :].squeeze()
            model.state_dict()['model.121.m.2.weight'].data[i, :, :, :] *= state_dict['model.121.im.2.implicit'].data[:, i, : :].squeeze()
        model.state_dict()['model.121.m.0.bias'].data += state_dict['model.121.m.0.weight'].mul(state_dict['model.121.ia.0.implicit']).sum(1).squeeze()
        model.state_dict()['model.121.m.1.bias'].data += state_dict['model.121.m.1.weight'].mul(state_dict['model.121.ia.1.implicit']).sum(1).squeeze()
        model.state_dict()['model.121.m.2.bias'].data += state_dict['model.121.m.2.weight'].mul(state_dict['model.121.ia.2.implicit']).sum(1).squeeze()
        model.state_dict()['model.121.m.0.bias'].data *= state_dict['model.121.im.0.implicit'].data.squeeze()
        model.state_dict()['model.121.m.1.bias'].data *= state_dict['model.121.im.1.implicit'].data.squeeze()
        model.state_dict()['model.121.m.2.bias'].data *= state_dict['model.121.im.2.implicit'].data.squeeze()

    elif opt.name == 'yolov7-w6':
        idx = 118
        idx2 = 122

        # check NetsPresso FD
        for i in state_dict:
            assert (f'model.{idx2}.m' in i and 'netspressofds' in i) == False, 'Reparameterization is not possible after using NetsPresso FD because the model structure has changed'
            
        # copy intersect weights
        exclude = []
        check_layer = []

        for k, v in state_dict.items():
            if k in model.state_dict() and not any(x in k for x in exclude):
                name_k = k.split('.')
                
                if ''.join(name_k[1:-1]) not in check_layer:
                    check_layer.append(''.join(name_k[1:-1]))
                    
                    if len(name_k) == 4:
                        if isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]], torch.nn.modules.conv.Conv2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]].in_channels = v.size(1)
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]].out_channels = v.size(0)
                        elif isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]], torch.nn.modules.batchnorm.BatchNorm2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]].num_features = v.size(0)
                    
                    elif len(name_k) == 5:
                        if isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]], torch.nn.modules.conv.Conv2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]].in_channels = v.size(1)
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]].out_channels = v.size(0)
                        elif isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]], torch.nn.modules.batchnorm.BatchNorm2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]].num_features = v.size(0)

        for k, v in state_dict.items():
            if k in model.state_dict() and not any(x in k for x in exclude):
                name_k = k.split('.')
                if v.dtype == torch.int64:
                    apply_v = torch.tensor(v)
                else:
                    apply_v = nn.Parameter(v)
                
                if len(name_k) == 4:
                    setattr(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]], name_k[3], apply_v) 
                elif len(name_k) == 5:
                    setattr(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]], name_k[4], apply_v)

        model.names = ckpt['model'].names
        model.nc = ckpt['model'].nc

        # copy weights of lead head
        model._modules['model']._modules[str(idx)]._modules['m']._modules['0'].in_channel = state_dict['model.{}.m.0.weight'.format(idx2)].size(1)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['0'].out_channel = state_dict['model.{}.m.0.weight'.format(idx2)].size(0)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['0'].weight.data = state_dict['model.{}.m.0.weight'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['0'].bias.data = state_dict['model.{}.m.0.bias'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['1'].in_channel = state_dict['model.{}.m.1.weight'.format(idx2)].size(1)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['1'].out_channel = state_dict['model.{}.m.1.weight'.format(idx2)].size(0)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['1'].weight.data = state_dict['model.{}.m.1.weight'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['1'].bias.data = state_dict['model.{}.m.1.bias'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['2'].in_channel = state_dict['model.{}.m.2.weight'.format(idx2)].size(1)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['2'].out_channel = state_dict['model.{}.m.2.weight'.format(idx2)].size(0)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['2'].weight.data = state_dict['model.{}.m.2.weight'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['2'].bias.data = state_dict['model.{}.m.2.bias'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['3'].in_channel = state_dict['model.{}.m.3.weight'.format(idx2)].size(1)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['3'].out_channel = state_dict['model.{}.m.3.weight'.format(idx2)].size(0)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['3'].weight.data = state_dict['model.{}.m.3.weight'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['3'].bias.data = state_dict['model.{}.m.3.bias'.format(idx2)].data

        # reparametrized YOLOR
        for i in range((model.nc+5)*anchors):
            model.state_dict()['model.{}.m.0.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.0.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.1.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.1.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.2.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.2.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.3.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.3.implicit'.format(idx2)].data[:, i, : :].squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].mul(state_dict['model.{}.ia.0.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].mul(state_dict['model.{}.ia.1.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].mul(state_dict['model.{}.ia.2.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].mul(state_dict['model.{}.ia.3.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data *= state_dict['model.{}.im.0.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data *= state_dict['model.{}.im.1.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data *= state_dict['model.{}.im.2.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data *= state_dict['model.{}.im.3.implicit'.format(idx2)].data.squeeze()

    elif opt.name == 'yolov7-e6':
        idx = 140
        idx2 = 144

        # check NetsPresso FD
        for i in state_dict:
            assert (f'model.{idx2}.m' in i and 'netspressofds' in i) == False, 'Reparameterization is not possible after using NetsPresso FD because the model structure has changed'
            
        # copy intersect weights
        exclude = []
        check_layer = []

        for k, v in state_dict.items():
            if k in model.state_dict() and not any(x in k for x in exclude):
                name_k = k.split('.')
                if name_k[1] == idx:
                    break
                
                if ''.join(name_k[1:-1]) not in check_layer:
                    check_layer.append(''.join(name_k[1:-1]))
                    
                    if len(name_k) == 4:
                        if isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]], torch.nn.modules.conv.Conv2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]].in_channels = v.size(1)
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]].out_channels = v.size(0)
                        elif isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]], torch.nn.modules.batchnorm.BatchNorm2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]].num_features = v.size(0)
                    
                    elif len(name_k) == 5:
                        if isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]], torch.nn.modules.conv.Conv2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]].in_channels = v.size(1)
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]].out_channels = v.size(0)
                        elif isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]], torch.nn.modules.batchnorm.BatchNorm2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]].num_features = v.size(0)

        for k, v in state_dict.items():
            if k in model.state_dict() and not any(x in k for x in exclude):
                name_k = k.split('.')
                if name_k[1] == idx:
                    break

                if v.dtype == torch.int64:
                    apply_v = torch.tensor(v)
                else:
                    apply_v = nn.Parameter(v)
                
                if len(name_k) == 4:
                    setattr(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]], name_k[3], apply_v) 
                elif len(name_k) == 5:
                    setattr(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]], name_k[4], apply_v)

        model.names = ckpt['model'].names
        model.nc = ckpt['model'].nc

        # copy weights of lead head
        model._modules['model']._modules[str(idx)]._modules['m']._modules['0'].in_channel = state_dict['model.{}.m.0.weight'.format(idx2)].size(1)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['0'].out_channel = state_dict['model.{}.m.0.weight'.format(idx2)].size(0)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['0'].weight.data = state_dict['model.{}.m.0.weight'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['0'].bias.data = state_dict['model.{}.m.0.bias'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['1'].in_channel = state_dict['model.{}.m.1.weight'.format(idx2)].size(1)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['1'].out_channel = state_dict['model.{}.m.1.weight'.format(idx2)].size(0)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['1'].weight.data = state_dict['model.{}.m.1.weight'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['1'].bias.data = state_dict['model.{}.m.1.bias'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['2'].in_channel = state_dict['model.{}.m.2.weight'.format(idx2)].size(1)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['2'].out_channel = state_dict['model.{}.m.2.weight'.format(idx2)].size(0)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['2'].weight.data = state_dict['model.{}.m.2.weight'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['2'].bias.data = state_dict['model.{}.m.2.bias'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['3'].in_channel = state_dict['model.{}.m.3.weight'.format(idx2)].size(1)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['3'].out_channel = state_dict['model.{}.m.3.weight'.format(idx2)].size(0)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['3'].weight.data = state_dict['model.{}.m.3.weight'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['3'].bias.data = state_dict['model.{}.m.3.bias'.format(idx2)].data

        # reparametrized YOLOR
        for i in range((model.nc+5)*anchors):
            model.state_dict()['model.{}.m.0.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.0.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.1.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.1.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.2.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.2.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.3.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.3.implicit'.format(idx2)].data[:, i, : :].squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].mul(state_dict['model.{}.ia.0.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].mul(state_dict['model.{}.ia.1.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].mul(state_dict['model.{}.ia.2.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].mul(state_dict['model.{}.ia.3.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data *= state_dict['model.{}.im.0.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data *= state_dict['model.{}.im.1.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data *= state_dict['model.{}.im.2.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data *= state_dict['model.{}.im.3.implicit'.format(idx2)].data.squeeze()

    elif opt.name == 'yolov7-d6':
        idx = 162
        idx2 = 166

        # check NetsPresso FD
        for i in state_dict:
            assert (f'model.{idx2}.m' in i and 'netspressofds' in i) == False, 'Reparameterization is not possible after using NetsPresso FD because the model structure has changed'
            
        # copy intersect weights
        exclude = []
        check_layer = []

        for k, v in state_dict.items():
            if k in model.state_dict() and not any(x in k for x in exclude):
                name_k = k.split('.')
                if name_k[1] == idx:
                    break
                
                if ''.join(name_k[1:-1]) not in check_layer:
                    check_layer.append(''.join(name_k[1:-1]))
                    
                    if len(name_k) == 4:
                        if isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]], torch.nn.modules.conv.Conv2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]].in_channels = v.size(1)
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]].out_channels = v.size(0)
                        elif isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]], torch.nn.modules.batchnorm.BatchNorm2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]].num_features = v.size(0)
                    
                    elif len(name_k) == 5:
                        if isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]], torch.nn.modules.conv.Conv2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]].in_channels = v.size(1)
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]].out_channels = v.size(0)
                        elif isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]], torch.nn.modules.batchnorm.BatchNorm2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]].num_features = v.size(0)

        for k, v in state_dict.items():
            if k in model.state_dict() and not any(x in k for x in exclude):
                name_k = k.split('.')
                if name_k[1] == idx:
                    break

                if v.dtype == torch.int64:
                    apply_v = torch.tensor(v)
                else:
                    apply_v = nn.Parameter(v)
                
                if len(name_k) == 4:
                    setattr(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]], name_k[3], apply_v) 
                elif len(name_k) == 5:
                    setattr(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]], name_k[4], apply_v)
        
        model.names = ckpt['model'].names
        model.nc = ckpt['model'].nc

        # copy weights of lead head
        model._modules['model']._modules[str(idx)]._modules['m']._modules['0'].in_channel = state_dict['model.{}.m.0.weight'.format(idx2)].size(1)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['0'].out_channel = state_dict['model.{}.m.0.weight'.format(idx2)].size(0)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['0'].weight.data = state_dict['model.{}.m.0.weight'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['0'].bias.data = state_dict['model.{}.m.0.bias'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['1'].in_channel = state_dict['model.{}.m.1.weight'.format(idx2)].size(1)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['1'].out_channel = state_dict['model.{}.m.1.weight'.format(idx2)].size(0)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['1'].weight.data = state_dict['model.{}.m.1.weight'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['1'].bias.data = state_dict['model.{}.m.1.bias'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['2'].in_channel = state_dict['model.{}.m.2.weight'.format(idx2)].size(1)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['2'].out_channel = state_dict['model.{}.m.2.weight'.format(idx2)].size(0)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['2'].weight.data = state_dict['model.{}.m.2.weight'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['2'].bias.data = state_dict['model.{}.m.2.bias'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['3'].in_channel = state_dict['model.{}.m.3.weight'.format(idx2)].size(1)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['3'].out_channel = state_dict['model.{}.m.3.weight'.format(idx2)].size(0)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['3'].weight.data = state_dict['model.{}.m.3.weight'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['3'].bias.data = state_dict['model.{}.m.3.bias'.format(idx2)].data

        # reparametrized YOLOR
        for i in range((model.nc+5)*anchors):
            model.state_dict()['model.{}.m.0.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.0.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.1.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.1.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.2.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.2.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.3.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.3.implicit'.format(idx2)].data[:, i, : :].squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].mul(state_dict['model.{}.ia.0.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].mul(state_dict['model.{}.ia.1.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].mul(state_dict['model.{}.ia.2.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].mul(state_dict['model.{}.ia.3.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data *= state_dict['model.{}.im.0.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data *= state_dict['model.{}.im.1.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data *= state_dict['model.{}.im.2.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data *= state_dict['model.{}.im.3.implicit'.format(idx2)].data.squeeze()

    elif opt.name == 'yolov7-e6e':
        idx = 261
        idx2 = 265

        # check NetsPresso FD
        for i in state_dict:
            assert (f'model.{idx2}.m' in i and 'netspressofds' in i) == False, 'Reparameterization is not possible after using NetsPresso FD because the model structure has changed'
            
        # copy intersect weights
        exclude = []
        check_layer = []

        for k, v in state_dict.items():
            if k in model.state_dict() and not any(x in k for x in exclude):
                name_k = k.split('.')
                if name_k[1] == idx:
                    break
                
                if ''.join(name_k[1:-1]) not in check_layer:
                    check_layer.append(''.join(name_k[1:-1]))
                    
                    if len(name_k) == 4:
                        if isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]], torch.nn.modules.conv.Conv2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]].in_channels = v.size(1)
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]].out_channels = v.size(0)
                        elif isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]], torch.nn.modules.batchnorm.BatchNorm2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]].num_features = v.size(0)
                    
                    elif len(name_k) == 5:
                        if isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]], torch.nn.modules.conv.Conv2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]].in_channels = v.size(1)
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]].out_channels = v.size(0)
                        elif isinstance(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]], torch.nn.modules.batchnorm.BatchNorm2d):
                            model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]].num_features = v.size(0)

        for k, v in state_dict.items():
            if k in model.state_dict() and not any(x in k for x in exclude):
                name_k = k.split('.')
                if name_k[1] == idx:
                    break

                if v.dtype == torch.int64:
                    apply_v = torch.tensor(v)
                else:
                    apply_v = nn.Parameter(v)
                
                if len(name_k) == 4:
                    setattr(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]], name_k[3], apply_v) 
                elif len(name_k) == 5:
                    setattr(model._modules[name_k[0]]._modules[name_k[1]]._modules[name_k[2]]._modules[name_k[3]], name_k[4], apply_v)
        
        model.names = ckpt['model'].names
        model.nc = ckpt['model'].nc

        # copy weights of lead head
        model._modules['model']._modules[str(idx)]._modules['m']._modules['0'].in_channel = state_dict['model.{}.m.0.weight'.format(idx2)].size(1)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['0'].out_channel = state_dict['model.{}.m.0.weight'.format(idx2)].size(0)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['0'].weight.data = state_dict['model.{}.m.0.weight'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['0'].bias.data = state_dict['model.{}.m.0.bias'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['1'].in_channel = state_dict['model.{}.m.1.weight'.format(idx2)].size(1)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['1'].out_channel = state_dict['model.{}.m.1.weight'.format(idx2)].size(0)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['1'].weight.data = state_dict['model.{}.m.1.weight'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['1'].bias.data = state_dict['model.{}.m.1.bias'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['2'].in_channel = state_dict['model.{}.m.2.weight'.format(idx2)].size(1)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['2'].out_channel = state_dict['model.{}.m.2.weight'.format(idx2)].size(0)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['2'].weight.data = state_dict['model.{}.m.2.weight'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['2'].bias.data = state_dict['model.{}.m.2.bias'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['3'].in_channel = state_dict['model.{}.m.3.weight'.format(idx2)].size(1)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['3'].out_channel = state_dict['model.{}.m.3.weight'.format(idx2)].size(0)
        model._modules['model']._modules[str(idx)]._modules['m']._modules['3'].weight.data = state_dict['model.{}.m.3.weight'.format(idx2)].data
        model._modules['model']._modules[str(idx)]._modules['m']._modules['3'].bias.data = state_dict['model.{}.m.3.bias'.format(idx2)].data

        # reparametrized YOLOR
        for i in range((model.nc+5)*anchors):
            model.state_dict()['model.{}.m.0.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.0.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.1.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.1.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.2.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.2.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.3.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.3.implicit'.format(idx2)].data[:, i, : :].squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].mul(state_dict['model.{}.ia.0.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].mul(state_dict['model.{}.ia.1.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].mul(state_dict['model.{}.ia.2.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].mul(state_dict['model.{}.ia.3.implicit'.format(idx2)]).sum(1).squeeze()
        model.state_dict()['model.{}.m.0.bias'.format(idx)].data *= state_dict['model.{}.im.0.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.1.bias'.format(idx)].data *= state_dict['model.{}.im.1.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.2.bias'.format(idx)].data *= state_dict['model.{}.im.2.implicit'.format(idx2)].data.squeeze()
        model.state_dict()['model.{}.m.3.bias'.format(idx)].data *= state_dict['model.{}.im.3.implicit'.format(idx2)].data.squeeze()

    # model to be saved
    ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
            'optimizer': None,
            'training_results': None,
            'epoch': -1}
    
    return ckpt

def parse_args():
    parser = argparse.ArgumentParser()

    """
        Common arguments
    """
    parser.add_argument('-n', '--name', type=str, default='yolov7', help='model name')
    parser.add_argument('-w', '--weights', type=str, default='./yolov7.pt', help='weights path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--load_netspresso', action='store_true', help='compress the compressed model')

    """
        Compression arguments
    """
    parser.add_argument("--compression_method", type=str, choices=["PR_L2", "PR_GM", "PR_NN", "PR_ID", "FD_TK", "FD_CP", "FD_SVD"], default="PR_L2")
    parser.add_argument("--recommendation_method", type=str, choices=["slamp", "vbmf"], default="slamp")
    parser.add_argument("--compression_ratio", type=int, default=0.5)
    parser.add_argument("-m", "--np_email", help="NetsPresso login e-mail", type=str)
    parser.add_argument("-p", "--np_password", help="NetsPresso login password", type=str)

    """
        Fine-tuning arguments
    """
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')

    """
        Export arguments
    """
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--dynamic-batch', action='store_true', help='dynamic batch onnx for tensorrt and onnx-runtime')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx')
    parser.add_argument('--max-wh', type=int, default=None, help='None for tensorrt nms, int value for onnx-runtime nms')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='conf threshold for NMS')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--include-nms', action='store_true', help='export end2end onnx')
    parser.add_argument('--export-device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_args()

    data = opt.data
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict

    # YOLOv7: 105, YOLOv7x: 121, YOLOv7-W6: 122, YOLOv7-E6:144, YOLOv7-D6: 166, YOLOv7-E6E: 265
    detect = {'yolov7': 105, 'yolov7x': 121, 'yolov7-w6': 122, 'yolov7-e6': 144, 'yolov7-d6': 166, 'yolov7-e6e': 265}
    assert opt.name in detect.keys()
    detect = str(detect[opt.name])

    """ 
        Convert YOLOv7 model to fx 
    """
    logger.info("yolov7 to fx graph start.")
    
    load_netspresso = opt.load_netspresso
    weights = opt.weights
     
    if load_netspresso: # after compression, the shape of the yaml file and the model do not match
        ckpt = torch.load(weights, map_location='cpu')
        model = ckpt['model'].float()
    else:
        nc = int(data_dict['nc'])  # number of classes
        ckpt = torch.load(weights, map_location='cpu')
        model = Model(ckpt['model'].yaml, ch=3, nc=nc)
        state_dict = ckpt['model'].float().state_dict()

        exclude = []
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
        model.load_state_dict(state_dict, strict=False)

    model.train()
    _graph = fx.Tracer().trace(model, {'augment': False, 'profile':False})
    traced_model = fx.GraphModule(model, _graph)
    torch.save(traced_model, f"{opt.name}_fx.pt")

    logger.info("yolov7 to fx graph end.")

    """
        Model compression - recommendation compression 
    """
    logger.info("Compression step start.")
    
    compressor = ModelCompressor(email=opt.np_email, password=opt.np_password)

    UPLOAD_MODEL_NAME = opt.name
    TASK = Task.OBJECT_DETECTION
    FRAMEWORK = Framework.PYTORCH
    UPLOAD_MODEL_PATH = f'{opt.name}_fx.pt'
    INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": opt.img_size}]
    model = compressor.upload_model(
        model_name=UPLOAD_MODEL_NAME,
        task=TASK,
        framework=FRAMEWORK,
        file_path=UPLOAD_MODEL_PATH,
        input_shapes=INPUT_SHAPES,
    )

    COMPRESSION_METHOD = opt.compression_method
    RECOMMENDATION_METHOD = opt.recommendation_method
    RECOMMENDATION_RATIO = opt.compression_ratio
    COMPRESSED_MODEL_NAME = f'{UPLOAD_MODEL_NAME}_{COMPRESSION_METHOD}_{RECOMMENDATION_RATIO}'.lower()
    OUTPUT_PATH = COMPRESSED_MODEL_NAME + '.pt'
    compressed_model = compressor.recommendation_compression(
        model_id=model.model_id,
        model_name=COMPRESSED_MODEL_NAME,
        compression_method=COMPRESSION_METHOD,
        recommendation_method=RECOMMENDATION_METHOD,
        recommendation_ratio=RECOMMENDATION_RATIO,
        output_path=OUTPUT_PATH,
    )

    logger.info("Compression step end.")

    """
        Retrain YOLOv7 model
    """
    logger.info("Fine-tuning step start.")

    opt.original = opt.weights
    opt.compressed = OUTPUT_PATH
    opt.detect = detect
    pt_file = fx2p(opt)

    torch.save(pt_file, COMPRESSED_MODEL_NAME + '_fx2p.pth')

    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)
        hyp['lr0'] *= 0.1
    
    with open('tmp_hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f)
    opt.hyp = 'tmp_hyp.yaml'

    opt.weights = COMPRESSED_MODEL_NAME + '_fx2p.pth'
    opt.netspresso = True
    opt = train_run(opt)

    logger.info("Fine-tuning step end.")

    """
        Export YOLOv5 model to onnx
    """
    logger.info("Export model to onnx format step start.")

    opt.weights = opt.save_dir + '/weights/best.pt'
    if opt.compression_method.split('_')[0] == 'PR': # FD cannot use reparameterization
        model = reparam(opt)
        torch.save(model, COMPRESSED_MODEL_NAME + '_before_onnx.pt')
        opt.weights = COMPRESSED_MODEL_NAME + '_before_onnx.pt'

    export_onnx(opt, COMPRESSED_MODEL_NAME + '.onnx')
    
    logger.info(f'=> saving model to {COMPRESSED_MODEL_NAME}.onnx')

    logger.info("Export model to onnx format step end.")
