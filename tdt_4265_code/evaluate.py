from utils import join_split_grids
from skimage import io
from sklearn.metrics import average_precision_score, precision_recall_curve,  PrecisionRecallDisplay
import torch
from torch import nn
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np
from resnet import ResNet, ResNetEnsembleInfraredRGB
from dataloaders import  load_sheep_grid_data, load_sheep_grid_multiband,load_sheep_grid_test_data, load_sheep_grid_multiband_test_data
from utils import to_cuda, get_grid, str2bool
from eval_utils import get_ensemble_model, do_predictions, write_results_to_file_and_calculate_AP
import time
import argparse
import json


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_type", default='rgb', type=str, help="rgb or infrared")
    parser.add_argument("--time_stamp", default='-', type=str, help="rgb or infrared")
    parser.add_argument("--batch_size", default=1, type=int, help="number of images in a batch")
    parser.add_argument("--rgb_resize_shape", default=0, type=int, help="size to resize rgb crop to") 
    parser.add_argument("--infrared_resize_shape", default=0, type=int, help="size to resize infrared crop to")
    parser.add_argument("--test_dataset", default='val', type=str, help="val, train or both... which dataset to evaluate") 
    parser.add_argument("--time_stamp_rgb", default='-', type=str, help="timestamp or rgb model to use")
    parser.add_argument("--time_stamp_infrared", default='-', type=str, help="timestamp or infrared model to use")
    parser.add_argument("--fuse_depth", default=3, type=int, help="level in network to fuse rgb and infrared (2,3,4 or 5")
    parser.add_argument("--network_depth", default=18, type=int, help="resnet depth (18, 34, 50, 101, 152)") 
    parser.add_argument("--X", default=False, type=str2bool, help="True if should use ResNeXt instead of ResNet") 

    args = parser.parse_args()


    root_work_dir = './Work_dirs/work_dirs_external/'
    work_dir = root_work_dir+ args.img_type +'/' + args.time_stamp

## Load Model ##
    if args.img_type == 'ensemble':
        model = get_ensemble_model(root_work_dir, args.time_stamp_rgb, args.time_stamp_infrared, args.network_depth, args.X, args.rgb_resize_shape, args.infrared_resize_shape, args.fuse_depth)

    else:
        model = ResNet(image_channels=3, num_classes=9, depth=args.network_depth, test_mode=True, X=args.X)

    model.cuda()

    pretrained_dict = torch.load(os.path.join(work_dir, 'model_best.pth.tar'))['state_dict'] 
    model.load_state_dict(pretrained_dict, strict=False)  
    #model = nn.DataParallel(model, device_ids = [0,1])
    model = model.eval()

    
## Load Dataset
    labels_val_path = 'annotations/val2020_simple.json'
    labels_train_path = 'annotations/train2020_simple.json'
    labels_test_path = 'annotations/test2020_simple.json'

    if args.img_type == 'ensemble':    
        dataloader_train, dataloader_val = load_sheep_grid_multiband(args.batch_size,
                                                     rgb_resize_shape= (args.rgb_resize_shape,args.rgb_resize_shape),
                                                     infrared_resize_shape=(args.infrared_resize_shape,args.infrared_resize_shape),
                                                     test_mode=True)
    else:
        dataloader_train, dataloader_val = load_sheep_grid_data(args.batch_size,
                                    img_type=args.img_type,
                                    test_mode=True,
                                    labels_val_path=labels_val_path,
                                    labels_train_path=labels_train_path,
                                    rgb_resize_shape= (args.rgb_resize_shape,args.rgb_resize_shape),
                                    infrared_resize_shape=(args.infrared_resize_shape,args.infrared_resize_shape))



## DO PREDICTIONS VALIDATION    
    if args.test_dataset in ['val','both','all']:
        results_val, gt_val, total_time_val = do_predictions(dataloader_val, args.img_type, model)
        print()
        print('======VALIDATION==========')
        print('TOTAL TIME: ', total_time_val)
        print('TIME per im: ', total_time_val/len(results_val.keys()))

        result_filename = 'validation_predictions_'+ args.img_type + '_' + str(args.time_stamp)   +'.txt'
        
        average_precision_val = write_results_to_file_and_calculate_AP(results_val, labels_val_path, work_dir, result_filename)
        print('Average precision-recall score : {0:0.3f}'.format(
average_precision_val))

## DO PREDICTIONS TRAIN   
    if args.test_dataset  in ['train','both','all']:
        results_train, gt_train, total_time_train = do_predictions(dataloader_train, args.img_type, model)
        print()
        print('======TRAIN==========')
        print('TOTAL TIME: ', total_time_train)
        print('TIME per im: ', total_time_train/len(results_train.keys()))
        result_filename = 'train_predictions_'+ args.img_type + '_' + str(args.time_stamp)   +'.txt'
        average_precision_train = write_results_to_file_and_calculate_AP(results_train, labels_train_path, work_dir, result_filename)
        print('Average precision-recall score : {0:0.3f}'.format(
average_precision_train))
        
## DO PREDICTIONS TEST   
    if args.test_dataset  in ['test','both','all']:

        if args.img_type == 'ensemble':    
            dataloader_test = load_sheep_grid_multiband_test_data(args.batch_size,
                                                         rgb_resize_shape= (args.rgb_resize_shape,args.rgb_resize_shape),
                                                         infrared_resize_shape=(args.infrared_resize_shape,args.infrared_resize_shape),
                                                         test_mode=True)
        else:
            dataloader_test = load_sheep_grid_test_data(args.batch_size,
                                        img_type=args.img_type,
                                        test_mode=True,
                                        rgb_resize_shape= (args.rgb_resize_shape,args.rgb_resize_shape),
                                        infrared_resize_shape=(args.infrared_resize_shape,args.infrared_resize_shape))
        results_test, gt_test, total_time_test = do_predictions(dataloader_test, args.img_type, model)
        print()
        print('======TEST==========')
        print('TOTAL TIME: ', total_time_test)
        print('TIME per im: ', total_time_test/len(results_test.keys()))
        result_filename = 'test_predictions_'+ args.img_type + '_' + str(args.time_stamp)   +'.txt'
        average_precision_test = write_results_to_file_and_calculate_AP(results_test, labels_test_path, work_dir, result_filename)
        print('Average precision-recall score : {0:0.3f}'.format(
average_precision_test))

# Append results to main results file
    if args.test_dataset == 'both':

        All_results_filename = 'results2.txt'

        with open(All_results_filename, "a") as file:
            file.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(args.time_stamp,
                        args.img_type,
                        str(args.fuse_depth) if args.img_type == 'ensemble' else '-',
                        str(args.network_depth),
                        str(args.X),
                        str(average_precision_val),
                        str(average_precision_train),
                        '-' if args.img_type == 'infrared' else str(args.rgb_resize_shape),
                        '-' if args.img_type == 'rgb' else str(args.infrared_resize_shape),
                        args.time_stamp_rgb,
                        args.time_stamp_infrared,
                        str(total_time_val/len(results_val.keys())),
                        str(total_time_train/len(results_train.keys())),
                        str(average_precision_test),
                        str(total_time_test/len(results_test.keys())),
                                                                               
                                                        ))


    


        
    