import torch
from torch import nn

import models
import argparse

from models.common import *
from collections import OrderedDict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=str, default='./yolor-csp-c.pt', help='weights path')
    parser.add_argument('--compressed', type=str, default='./yolor-csp-c.pt', help='weights path')
    parser.add_argument('--detect', type=str, default='105', help='number of detect()') # YOLOv7: 105, YOLOv7x: 121, YOLOv7-W6: 122, YOLOv7-E6:144, YOLOv7-D6: 166, YOLOv7-E6E: 265
    opt = parser.parse_args()
    
    original_path = opt.original
    compressed_path = opt.compressed
    detect = opt.detect
    
    pt_file = torch.load(original_path)
    original_model = pt_file['model'].float()
    compressed_model = torch.load(compressed_path).float()
    
    compressed_module_list = sorted(list(module for module in compressed_model._modules['module_dict']._modules if 'module_dict_module_dict_' in module)) # sort list

    def chunk_list(module_list, chunk_size):
        result = []
        for i in range(0, len(module_list), chunk_size):
            result.append(module_list[i:i+chunk_size])
        return result

    def flatten_list(nested_list):
        flat_list = []
        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(flatten_list(item))
            else:
                flat_list.append(item)
        return flat_list

    def create_ordered_dict_from_list(flat_list):
        ordered_dict = OrderedDict()
        for i, item in enumerate(flat_list):
            key = str(i)
            ordered_dict[key] = item
        return ordered_dict

    if compressed_module_list[0].split('_')[-2] == "tucker":
        chunk_size = 3
        chunked_list = chunk_list(compressed_module_list, chunk_size)
        for unit_module in chunked_list:
            temp_modules = []
            for i in range(chunk_size):
                compressed_model._modules['module_dict']._modules[unit_module[i]].weight = nn.Parameter(compressed_model._modules['module_dict']._modules[unit_module[i]].weight.contiguous())
                temp_modules.append(compressed_model._modules['module_dict']._modules[unit_module[i]])
            
            hierarchical_list = unit_module[0].split('_')[4:-2]
            if "netspressofds" not in unit_module[0]:
                temp_layer = Netspresso_FD(temp_modules)
            
                if len(hierarchical_list) == 3:
                    original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]] = temp_layer
                elif len(hierarchical_list) == 4:
                    original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]] = temp_layer
                elif len(hierarchical_list) == 5:
                    temp = hierarchical_list[2] + '_' + hierarchical_list[3]
                    original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]] = temp_layer
            else:
                if len(hierarchical_list) == 5:
                    temp_list = list(original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules.values())
                    temp_list[int(hierarchical_list[4])] = temp_modules
                    temp_list = create_ordered_dict_from_list(flatten_list(temp_list))
                    original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules = temp_list
                    
                elif len(hierarchical_list) == 6:
                    temp_list = list(original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]]._modules.values())
                    temp_list[int(hierarchical_list[5])] = temp_modules
                    temp_list = create_ordered_dict_from_list(flatten_list(temp_list))
                    original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]]._modules = temp_list
                    
                elif len(hierarchical_list) == 7:
                    temp = hierarchical_list[2] + '_' + hierarchical_list[3]
                    temp_list = list(original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]]._modules.values())
                    temp_list[int(hierarchical_list[6])] = temp_modules
                    temp_list = create_ordered_dict_from_list(flatten_list(temp_list))
                    original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]]._modules = temp_list

    elif compressed_module_list[0].split('_')[-2] == "svd":
        chunk_size = 2
        chunked_list = chunk_list(compressed_module_list, chunk_size)
        for unit_module in chunked_list:
            temp_modules = []
            for i in range(chunk_size):
                compressed_model._modules['module_dict']._modules[unit_module[i]].weight = nn.Parameter(compressed_model._modules['module_dict']._modules[unit_module[i]].weight.contiguous())
                temp_modules.append(compressed_model._modules['module_dict']._modules[unit_module[i]])
            
            hierarchical_list = unit_module[0].split('_')[4:-2] 
            if "netspressofds" not in unit_module[0]:
                temp_layer = Netspresso_FD(temp_modules)

                if len(hierarchical_list) == 3:
                    original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]] = temp_layer
                elif len(hierarchical_list) == 4:
                    original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]] = temp_layer
                elif len(hierarchical_list) == 5:
                    temp = hierarchical_list[2] + '_' + hierarchical_list[3]
                    original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]] = temp_layer
            else:
                if len(hierarchical_list) == 5:
                    temp_list = list(original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules.values())
                    temp_list[int(hierarchical_list[4])] = temp_modules
                    temp_list = create_ordered_dict_from_list(flatten_list(temp_list))
                    original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules = temp_list
                    
                elif len(hierarchical_list) == 6:
                    temp_list = list(original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]]._modules.values())
                    temp_list[int(hierarchical_list[5])] = temp_modules
                    temp_list = create_ordered_dict_from_list(flatten_list(temp_list))
                    original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]]._modules = temp_list
                    
                elif len(hierarchical_list) == 7:
                    temp = hierarchical_list[2] + '_' + hierarchical_list[3]
                    temp_list = list(original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]]._modules.values())
                    temp_list[int(hierarchical_list[6])] = temp_modules
                    temp_list = create_ordered_dict_from_list(flatten_list(temp_list))
                    original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]]._modules = temp_list
    
    else:
        for module in compressed_model._modules['module_dict']._modules:
            if 'module_dict_module_dict_' in module:
                hierarchical_list = module.split('_')[4:]
                with torch.no_grad():
                    compressed_model._modules['module_dict']._modules[module]._parameters['weight'] = compressed_model._modules['module_dict']._modules[module]._parameters['weight'].contiguous()
                if len(hierarchical_list) == 3:
                    original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]].weight.data = compressed_model._modules['module_dict']._modules[module].weight.data
                    if original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]].bias != None:
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]].bias.data = compressed_model._modules['module_dict']._modules[module].bias.data
                    if isinstance(compressed_model._modules['module_dict']._modules[module], torch.nn.modules.conv.Conv2d):
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]].in_channels = compressed_model._modules['module_dict']._modules[module].in_channels
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]].out_channels = compressed_model._modules['module_dict']._modules[module].out_channels
                    elif isinstance(compressed_model._modules['module_dict']._modules[module], torch.nn.modules.batchnorm.BatchNorm2d):
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]].num_features = compressed_model._modules['module_dict']._modules[module].num_features
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._buffers['running_mean'] = compressed_model._modules['module_dict']._modules[module]._buffers['running_mean']
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._buffers['running_var'] = compressed_model._modules['module_dict']._modules[module]._buffers['running_var']
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._buffers['num_batches_tracked'] = compressed_model._modules['module_dict']._modules[module]._buffers['num_batches_tracked']
                        
                elif len(hierarchical_list) == 4:
                    original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]].weight.data = compressed_model._modules['module_dict']._modules[module].weight.data
                    if original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]].bias != None:
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]].bias.data = compressed_model._modules['module_dict']._modules[module].bias.data
                    if isinstance(compressed_model._modules['module_dict']._modules[module], torch.nn.modules.conv.Conv2d):
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]].in_channels = compressed_model._modules['module_dict']._modules[module].in_channels
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]].out_channels = compressed_model._modules['module_dict']._modules[module].out_channels
                    elif isinstance(compressed_model._modules['module_dict']._modules[module], torch.nn.modules.batchnorm.BatchNorm2d):
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._buffers['running_mean'] = compressed_model._modules['module_dict']._modules[module]._buffers['running_mean']
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._buffers['running_var'] = compressed_model._modules['module_dict']._modules[module]._buffers['running_var']
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._buffers['num_batches_tracked'] = compressed_model._modules['module_dict']._modules[module]._buffers['num_batches_tracked']
                        
                elif len(hierarchical_list) == 5:
                    if 'netspressofds' not in hierarchical_list:
                        temp = hierarchical_list[2] + '_' + hierarchical_list[3]
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]].weight.data = compressed_model._modules['module_dict']._modules[module].weight.data
                        if original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]].bias != None:
                            original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]].bias.data = compressed_model._modules['module_dict']._modules[module].bias.data
                        if isinstance(compressed_model._modules['module_dict']._modules[module], torch.nn.modules.conv.Conv2d):
                            original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]].in_channels = compressed_model._modules['module_dict']._modules[module].in_channels
                            original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]].out_channels = compressed_model._modules['module_dict']._modules[module].out_channels
                        elif isinstance(compressed_model._modules['module_dict']._modules[module], torch.nn.modules.batchnorm.BatchNorm2d):
                            original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]]._buffers['running_mean'] = compressed_model._modules['module_dict']._modules[module]._buffers['running_mean']
                            original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]]._buffers['running_var'] = compressed_model._modules['module_dict']._modules[module]._buffers['running_var']
                            original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]]._buffers['num_batches_tracked'] = compressed_model._modules['module_dict']._modules[module]._buffers['num_batches_tracked']
                    else:
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]].weight.data = compressed_model._modules['module_dict']._modules[module].weight.data
                        if original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]].bias != None:
                            original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]].bias.data = compressed_model._modules['module_dict']._modules[module].bias.data
                        if isinstance(compressed_model._modules['module_dict']._modules[module], torch.nn.modules.conv.Conv2d):
                            original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]].in_channels = compressed_model._modules['module_dict']._modules[module].in_channels
                            original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]].out_channels = compressed_model._modules['module_dict']._modules[module].out_channels
                        elif isinstance(compressed_model._modules['module_dict']._modules[module], torch.nn.modules.batchnorm.BatchNorm2d):
                            original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]]._buffers['running_mean'] = compressed_model._modules['module_dict']._modules[module]._buffers['running_mean']
                            original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]]._buffers['running_var'] = compressed_model._modules['module_dict']._modules[module]._buffers['running_var']
                            original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]]._buffers['num_batches_tracked'] = compressed_model._modules['module_dict']._modules[module]._buffers['num_batches_tracked']
                
                elif len(hierarchical_list) == 6:
                    original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]].weight.data = compressed_model._modules['module_dict']._modules[module].weight.data
                    if original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]].bias != None:
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]].bias.data = compressed_model._modules['module_dict']._modules[module].bias.data
                    if isinstance(compressed_model._modules['module_dict']._modules[module], torch.nn.modules.conv.Conv2d):
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]].in_channels = compressed_model._modules['module_dict']._modules[module].in_channels
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]].out_channels = compressed_model._modules['module_dict']._modules[module].out_channels
                    elif isinstance(compressed_model._modules['module_dict']._modules[module], torch.nn.modules.batchnorm.BatchNorm2d):
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]]._buffers['running_mean'] = compressed_model._modules['module_dict']._modules[module]._buffers['running_mean']
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]]._buffers['running_var'] = compressed_model._modules['module_dict']._modules[module]._buffers['running_var']
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[hierarchical_list[2]]._modules[hierarchical_list[3]]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]]._buffers['num_batches_tracked'] = compressed_model._modules['module_dict']._modules[module]._buffers['num_batches_tracked']
                
                elif len(hierarchical_list) == 7:
                    temp = hierarchical_list[2] + '_' + hierarchical_list[3]
                    original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]]._modules[hierarchical_list[6]].weight.data = compressed_model._modules['module_dict']._modules[module].weight.data
                    if original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]]._modules[hierarchical_list[6]].bias != None:
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]]._modules[hierarchical_list[6]].bias.data = compressed_model._modules['module_dict']._modules[module].bias.data
                    if isinstance(compressed_model._modules['module_dict']._modules[module], torch.nn.modules.conv.Conv2d):
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]]._modules[hierarchical_list[6]].in_channels = compressed_model._modules['module_dict']._modules[module].in_channels
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]]._modules[hierarchical_list[6]].out_channels = compressed_model._modules['module_dict']._modules[module].out_channels
                    elif isinstance(compressed_model._modules['module_dict']._modules[module], torch.nn.modules.batchnorm.BatchNorm2d):
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]]._modules[hierarchical_list[6]]._buffers['running_mean'] = compressed_model._modules['module_dict']._modules[module]._buffers['running_mean']
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]]._modules[hierarchical_list[6]]._buffers['running_var'] = compressed_model._modules['module_dict']._modules[module]._buffers['running_var']
                        original_model._modules[hierarchical_list[0]]._modules[hierarchical_list[1]]._modules[temp]._modules[hierarchical_list[4]]._modules[hierarchical_list[5]]._modules[hierarchical_list[6]]._buffers['num_batches_tracked'] = compressed_model._modules['module_dict']._modules[module]._buffers['num_batches_tracked']

    # ia            
    for module in compressed_model._modules['module_dict']._modules['module_dict']._modules['module_dict']._modules['model']._modules[detect]._modules['ia']._modules:
        original_model._modules['model']._modules[detect]._modules['ia']._modules[module]._parameters['implicit'].data = \
            compressed_model._modules['module_dict']._modules['module_dict']._modules['module_dict']._modules['model']._modules[detect]._modules['ia']._modules[module]._parameters['NOTA_implicit'].data
        original_model._modules['model']._modules[detect]._modules['ia']._modules[module].channel = \
            compressed_model._modules['module_dict']._modules['module_dict']._modules['module_dict']._modules['model']._modules[detect]._modules['ia']._modules[module]._parameters['NOTA_implicit'].size()[1]
    # im
    for module in compressed_model._modules['module_dict']._modules['module_dict']._modules['module_dict']._modules['model']._modules[detect]._modules['im']._modules:
        original_model._modules['model']._modules[detect]._modules['im']._modules[module]._parameters['implicit'].data = \
            compressed_model._modules['module_dict']._modules['module_dict']._modules['module_dict']._modules['model']._modules[detect]._modules['im']._modules[module]._parameters['NOTA_implicit'].data
        original_model._modules['model']._modules[detect]._modules['im']._modules[module].channel = \
            compressed_model._modules['module_dict']._modules['module_dict']._modules['module_dict']._modules['model']._modules[detect]._modules['im']._modules[module]._parameters['NOTA_implicit'].size()[1]
        
    # RepConv
    for module in original_model._modules['model']._modules:
        if isinstance(original_model._modules['model']._modules[module], (models.common.RepConv, models.common.RepConv_OREPA)) and not hasattr(original_model._modules['model']._modules[module]._modules['rbr_dense'][0], 'netspressofds'):
            original_model._modules['model']._modules[module].in_channels = original_model._modules['model']._modules[module]._modules['rbr_dense'][0].in_channels
            if hasattr(original_model._modules['model']._modules[module], 'out_channels'):
                original_model._modules['model']._modules[module].out_channels = original_model._modules['model']._modules[module]._modules['rbr_dense'][0].out_channels
        
    pt_file['model'] = original_model
    torch.save(pt_file, 'fx2p_complete.pt')
