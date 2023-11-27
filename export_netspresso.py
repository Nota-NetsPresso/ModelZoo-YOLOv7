from models.yolo import Model
import torch
import yaml
from utils.torch_utils import intersect_dicts, select_device
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolor-csp-c.pt', help='weights path')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--netspresso', action='store_true', help='compress the compressed model')
    opt = parser.parse_args()

    data = opt.data
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    
    netspresso = opt.netspresso
    weights = opt.weights
    device = select_device(opt.device)
     
    if netspresso: # after compression, the shape of the yaml file and the model do not match
        ckpt = torch.load(weights, map_location=device)
        model = ckpt['model'].float()
    else:
        nc = int(data_dict['nc'])  # number of classes
        ckpt = torch.load(weights, map_location=device)
        model = Model(ckpt['model'].yaml, ch=3, nc=nc).to(device)
        state_dict = ckpt['model'].float().state_dict()

        exclude = []
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
        model.load_state_dict(state_dict, strict=False)

    import torch.fx as fx

    model.train()
    _graph = fx.Tracer().trace(model)
    traced_model = fx.GraphModule(model, _graph)
    torch.save(traced_model, "yolov7_fx.pt")