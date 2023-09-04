import argparse

import yaml
from loguru import logger
import torch
import torch.fx as fx

from netspresso.compressor import ModelCompressor, Task, Framework

from utils.torch_utils import intersect_dicts
from models.yolo import Model


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
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')


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
    COMPRESSED_MODEL_NAME = f'{UPLOAD_MODEL_NAME}_{COMPRESSION_METHOD}_{RECOMMENDATION_RATIO}'
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
