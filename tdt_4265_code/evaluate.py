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
from dataloaders import  load_sheep_grid_data, load_sheep_grid_multiband
from utils import to_cuda
import time
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_type", default='rgb', type=str, help="rgb or infrared")
    parser.add_argument("--time_stamp", default='20200229_2120', type=str, help="rgb or infrared")
    parser.add_argument("--batch_size", default=1, type=int, help="number of images in a batch")
    parser.add_argument("--rgb_resize_shape", default=1280, type=int, help="size to resize rgb crop to") 
    parser.add_argument("--infrared_resize_shape", default=160, type=int, help="size to resize infrared crop to")
    parser.add_argument("--test_dataset", default='val', type=str, help="val or train... which dataset to evalutate") 
    parser.add_argument("--time_stamp_rgb", default='20200225_1439', type=str, help="timestamp or rgb model to use")
    parser.add_argument("--time_stamp_infrared", default='20200221_1432', type=str, help="timestamp or infrared model to use")
    parser.add_argument("--fuse_depth", default=3, type=int, help="level in network to fuse rgb and infrared (2,3,4 or 5") 

    args = parser.parse_args()

    

    root_work_dir = './Work_dirs/work_dirs_external/'
    work_dir = root_work_dir+ args.img_type +'/' + args.time_stamp
    
    ## IF ENSEMBLE ###
    time_stamp_rgb = '20200222_0826'
    time_stamp_infrared = '20200221_1432'

    if args.img_type == 'ensemble':    
        model_path_rgb = root_work_dir + 'rgb/' + args.time_stamp_rgb +'/model_best.pth.tar'
        model_rgb = ResNet(image_channels=3, num_classes=9)
        model_rgb.load_state_dict(torch.load(model_path_rgb)['state_dict'])

        model_path_infrared = root_work_dir + 'infrared/' + args.time_stamp_infrared  +'/model_best.pth.tar'
        model_infrared = ResNet(image_channels=3, num_classes=9)
        model_infrared.load_state_dict(torch.load(model_path_infrared)['state_dict'])


        model = ResNetEnsembleInfraredRGB(num_classes=9, ResNetRGB=model_rgb, ResNetIR=model_infrared,
                                         rgb_size = args.rgb_resize_shape,
                                         infrared_size = args.infrared_resize_shape,
                                         fuse_after_layer = args.fuse_depth)

    else:
        model = ResNet(image_channels=3, num_classes=9)


    pretrained_dict = torch.load(os.path.join(work_dir, 'model_best.pth.tar'), map_location="cuda:0")['state_dict']
    model.load_state_dict(pretrained_dict, strict=True)    

    model.cuda()
    model = model.eval()
    
    # Load our dataset
    labels_val_path = 'annotations/val2020_simple.json'
    labels_train_path = 'annotations/train2020_simple.json'

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
    #DO PREDICTIONS                                                        
    torch.cuda.empty_cache()
    all_pred_val = []
    all_gt_val = []


    results_split = {}
    gt_split = {}


    if args.test_dataset == 'train':
        dataloader = dataloader_train 
    else:
        dataloader = dataloader_val


    start = time.time()
    for batch_it, sample in enumerate(dataloader):    

        if args.img_type == 'ensemble':
            X_batch_rgb = sample['rgb']
            X_batch_infrared = sample['infrared']
            Y_batch = sample['grid']

            X_batch_rgb = to_cuda(X_batch_rgb, 0)
            X_batch_infrared = to_cuda(X_batch_infrared, 0)
            Y_batch = to_cuda(Y_batch, 0)

            ims = np.array(X_batch_rgb.permute(0,2,3,1).cpu().detach())
            ims2 = np.array(X_batch_infrared.permute(0,2,3,1).cpu().detach())

            # Perform the forward pass
            predictions = model(X_batch_rgb, X_batch_infrared)

        else:
            X_batch = to_cuda(sample['image'])
            Y_batch = to_cuda(sample['grid'])        
            predictions = model(X_batch)    


        Key_batch =  sample['key']
        predictions = torch.sigmoid(predictions)


        for i in range(len(Key_batch)):
            key = Key_batch[i]
            prediction = predictions[i]
            results_split[key] = {
                'grid': prediction.cpu().detach().numpy(),
                                 }

            gt_split[key] = {
                'grid': np.array(Y_batch[i].cpu().detach().numpy())
                                 }
    #JOIN SPLIT PREDICTIONS                                                            
    results_full_ims = join_split_grids(results_split, crop_shape=(1200,1200), im_shape=(2400,3200), grid_shape=(3,3))
    gt_full_ims = join_split_grids(gt_split, crop_shape=(1200,1200), im_shape=(2400,3200), grid_shape=(3,3))

    end = time.time()
    total_time = end - start
    print()
    print('TOTAL TIME: ', total_time)
    print('TIME per im: ', total_time/len(results_full_ims.keys()))
    all_pred_val = []
    all_gt_val = []


    if args.test_dataset == 'train':
        result_filename = 'train_predictions.txt'
        dataset_folder = 'train2020'
    else:
        result_filename = 'validation_predictions.txt'
        dataset_folder = 'val2020'

    with open(os.path.join(work_dir,result_filename), "w") as file:
                file.write("")


    i=0
    for full_im_key in results_full_ims.keys():

        all_pred_val = [*all_pred_val, *results_full_ims[full_im_key].flatten()]
        all_gt_val = [*all_gt_val, *gt_full_ims[full_im_key].flatten()]

        with open(os.path.join(work_dir,result_filename), "a") as file:
                file.write("{} {}\n".format(key, str(results_full_ims[full_im_key].flatten())[1:-1]))

        i = i+1
        
    #REMOVE fuzzy cases
    mask = np.array(all_gt_val)!= -1
    all_gt_val_filtered = np.array(all_gt_val.copy())[mask]
    all_pred_val_filtered = np.array(all_pred_val.copy())[mask]                                                            
    average_precision_filtered = average_precision_score(np.array(all_gt_val_filtered).astype(int), all_pred_val_filtered)
    print('Average precision-recall score FILTERED: {0:0.3f}'.format(
average_precision_filtered))


        
    