import argparse

import yaml
from loguru import logger
import torch
import torch.fx as fx

from utils.torch_utils import intersect_dicts
from models.yolo import Model


def parse_args():
    parser = argparse.ArgumentParser()

    """
        Common arguments
    """
    parser.add_argument('-w', '--weights', type=str, default='./yolov7.pt', help='weights path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--load_netspresso', action='store_true', help='compress the compressed model')

    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_args()

    data = opt.data
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict

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
    torch.save(traced_model, "yolov7_fx.pt")

    logger.info("yolov7 to fx graph end.")
