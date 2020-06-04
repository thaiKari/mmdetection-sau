import time
import torch
from utils import to_cuda, get_grid, join_split_grids
import numpy as np
import os
import json
from sklearn.metrics import average_precision_score
from resnet import ResNet, ResNetEnsembleInfraredRGB

def get_ensemble_model(root_work_dir, time_stamp_rgb, time_stamp_infrared, network_depth, X, rgb_size, infrared_size, fuse_depth):
    
    model_path_rgb = root_work_dir + 'rgb/' + time_stamp_rgb +'/model_best.pth.tar'
    model_rgb = ResNet(image_channels=3, num_classes=9, depth=network_depth, test_mode=True, X=X)
    model_rgb.load_state_dict(torch.load(model_path_rgb)['state_dict'])

    model_path_infrared = root_work_dir + 'infrared/' + time_stamp_infrared  +'/model_best.pth.tar'
    model_infrared = ResNet(image_channels=3, num_classes=9, depth=network_depth, test_mode=True, X=X)
    model_infrared.load_state_dict(torch.load(model_path_infrared)['state_dict'])


    model = ResNetEnsembleInfraredRGB(num_classes=9, ResNetRGB=model_rgb, ResNetIR=model_infrared,
                                     rgb_size = rgb_size,
                                     infrared_size = infrared_size,
                                     fuse_after_layer = fuse_depth,
                                     depth=network_depth,
                                     test_mode=True)

    return model


def do_predictions(dataloader, img_type, model):
    torch.cuda.empty_cache()
    all_pred_val = []
    all_gt_val = []


    results_split = {}
    gt_split = {}
    
    torch.cuda.synchronize()
    start = time.time()
    for batch_it, sample in enumerate(dataloader):

        if img_type == 'ensemble':
            
            X_batch_rgb = sample['rgb']
            X_batch_infrared = sample['infrared']
            Y_batch = sample['grid']

            X_batch_rgb = to_cuda(X_batch_rgb, 0)
            X_batch_infrared = to_cuda(X_batch_infrared, 0)
            Y_batch = to_cuda(Y_batch, 0)

            #ims = np.array(X_batch_rgb.permute(0,2,3,1).cpu().detach())
            #ims2 = np.array(X_batch_infrared.permute(0,2,3,1).cpu().detach())

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
    
    
    return results_full_ims, gt_full_ims, total_time


def write_results_to_file_and_calculate_AP(results_full_ims, labels_path, work_dir, result_filename):
    
    all_pred = []
    all_gt = []
    
    with open(os.path.join(work_dir,result_filename), "w") as file:
                file.write("")
            
        
    with open(os.path.join('./data/data_external/',labels_path),'r') as file:
        bbox_map = json.load(file)
    
    #i=0
    for full_im_key in results_full_ims.keys():

        all_pred = [*all_pred, *results_full_ims[full_im_key].flatten()]
        all_gt = [*all_gt, *get_grid(bbox_map[full_im_key], im_shape=(2400, 3200), grid_shape = (6, 8)).flatten()]

        with open(os.path.join(work_dir,result_filename), "a") as file:
            file.write("{} {}\n".format(full_im_key, str(list(results_full_ims[full_im_key].flatten()))[1:-1]).replace(',','') )

        #i = i+1
        
    #REMOVE fuzzy cases
    mask = np.array(all_gt)!= -1
    all_gt_filtered = np.array(all_gt.copy())[mask]
    all_pred_filtered = np.array(all_pred.copy())[mask]                                                            
    average_precision_filtered = average_precision_score(np.array(all_gt_filtered).astype(int), all_pred_filtered)
    
    return average_precision_filtered



    
