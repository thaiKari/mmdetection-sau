#Starter code from https://raw.githubusercontent.com/hukkelas/TDT4265-A3-starter-code/master/utils.py
import torch
import shutil
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import json
import math



def to_cuda(elements, which_gpu=0):
    """
    Transfers elements to GPU memory, if a nvidia- GPU is available.
    Args:
        elements: A list or a single pytorch module.
    Returns:
        The same list transferred to GPU memory
    """

    if which_gpu == 0:
        cuda = torch.device('cuda:0')
    elif which_gpu == 1:
        cuda = torch.device('cuda:1')
    
    if torch.cuda.is_available(): # Checks if a GPU is available for pytorch
        if isinstance(elements, (list, tuple)):
            return [x.cuda(cuda) for x in elements] # Transfer each index of the list to GPU memory
        return elements.cuda(cuda)
    return elements


def compute_loss_and_accuracy(dataloader, model, loss_criterion, which_gpu, ensemble_learning=False):
    """
    Computes the total loss and accuracy over the whole dataloader
    Args:
        dataloder: Validation/Test dataloader
        model: torch.nn.Module
        loss_criterion: The loss criterion, e.g: nn.CrossEntropyLoss()
    Returns:
        [loss_avg, accuracy]: both scalar.
    """
    # Tracking variables
    loss_avg = 0
    total_correct = 0
    total_images = 0
    total_steps = 0

    for sample in dataloader:
        if not ensemble_learning:
            X_batch = sample['image']
            Y_batch = sample['grid']

            # Transfer images/labels to GPU VRAM, if possible
            X_batch = to_cuda(X_batch, which_gpu=which_gpu)
            Y_batch = to_cuda(Y_batch, which_gpu=which_gpu)
            # Forward pass the images through our model
            output_probs = model(X_batch) 
            #Size output_probs: [batch_size, n_classes]
            #Size output_probs: [batch_size] (the class number for each image in the batch)
            # Compute loss
            
        else:
            X_batch_rgb = sample['rgb']
            X_batch_infrared = sample['infrared']
            Y_batch = sample['grid']

            X_batch_rgb = to_cuda(X_batch_rgb, which_gpu=which_gpu)
            X_batch_infrared = to_cuda(X_batch_infrared,which_gpu=which_gpu)
            Y_batch = to_cuda(Y_batch, which_gpu=which_gpu)

            # Perform the forward pass
            output_probs = model(X_batch_rgb, X_batch_infrared)
            
        
        #print('output_probs', output_probs.size())
        loss = loss_criterion(output_probs, Y_batch)
        #print('loss', loss)

        # Predicted class is the max index over the column dimension
        #predictions = output_probs.argmax(dim=1).squeeze()
        #Y_batch = Y_batch.squeeze()

        # Update tracking variables
        loss_avg += loss.item()
        total_steps += 1
        #total_correct += (predictions == Y_batch).sum().item()
        total_images += Y_batch.shape[0]
    loss_avg = loss_avg / total_steps
    #accuracy = total_correct / total_images
    return loss_avg#, accuracy


'''
save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
'''
def save_checkpoint(state, is_best, checkpoint_path='./', filename='checkpoint.pth.tar'):
    #Only save checkpoint if new best
    if is_best:
        torch.save(state, filename)
        print('saving state epoch', state['epoch'])
        print('new best!')
        shutil.copyfile(filename, os.path.join(checkpoint_path, 'model_best.pth.tar'))
        
        

        
'''
LOAD:
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
'''

def read_grid_labels(file_path):
    
    f = open(file_path, "r")
    result = {}
    for line in f:
        line_split = line.split()
        im_name = line_split[0]
        grid_vals = np.array(line_split[1:]).astype(int)
        result[im_name] = grid_vals

    return result

def read_coco_annotations(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        
    return data



#no overlap
def get_minxys(h, w, h_new, w_new):
    pos = 0
    minxs = []
    while pos < w - w_new:
        minxs.append(int(pos))
        pos = pos + w_new
    minxs.append(int(w-w_new))

    pos = 0
    minys = []
    while pos < h - h_new:
        minys.append(int(pos))
        pos = pos + h_new
    minys.append(int(h-h_new))
    
    return minxs, minys


def tensor_grid_to_2d_numpy_grid(tensor_grid, grid_shape):
    return tensor_grid.numpy().reshape(grid_shape)


def get_grid_minx_miny_from_split_im_key(key):
    key_split = key.split('_')
    grid_minx = key_split[-2]
    grid_miny = key_split[-1]
    return int(grid_minx), int(grid_miny)

def build_split_im_key(key, grid_minx, grid_miny ):
    return str(key) + '_' + str(grid_minx) + '_' + str(grid_miny)


def join_split_grids(split_grid_map, crop_shape=(1200,1200), im_shape=(2400,3200), grid_shape=(3,3)):
    
    full_grid_map = {}
    
    grid_h = crop_shape[0]/grid_shape[0]
    grid_w = crop_shape[1]/grid_shape[1]

    full_grid_shape = (int(im_shape[0]/grid_h), int(im_shape[1]/grid_w) )
    new_full_grid = np.zeros(full_grid_shape)

    #Go through each split key and use it to  build full grid
    for key in split_grid_map.keys():
        full_grid_key = key.split('.JPG')[0]+'.JPG'
        
        if full_grid_key in full_grid_map.keys():
            full_grid = full_grid_map[full_grid_key]        
        else:
            full_grid = new_full_grid.copy()

        
        grid_part  = split_grid_map[key]        
        grid_minx, grid_miny = get_grid_minx_miny_from_split_im_key(key)
        
        
        if type(grid_part['grid']) == type(np.zeros(0)):
            grid = grid_part['grid'].reshape(grid_shape)
        
        else:
            grid = tensor_grid_to_2d_numpy_grid(grid_part['grid'], grid_shape)

        for x in range(grid_shape[0]):
            for y in range(grid_shape[1]):

                grid_val = grid[y,x]
                grid_x = x + grid_minx
                grid_y = y + grid_miny

                if full_grid[grid_y, grid_x] == 0:
                    full_grid[grid_y, grid_x] = grid_val
                else:
                    full_grid[grid_y, grid_x] = (full_grid[grid_y, grid_x] + grid_val ) /2
         
        full_grid_map[full_grid_key] =  full_grid
        
    return full_grid_map
        


def get_new_bbox_coords_from_split_im(data, minx, miny, crop_shape):
    
    new_data =[]
    
    for x,y,w,h in data:        
        
        newxmin = x - minx
        newymin = y - miny        
        newxmax = newxmin + w
        newymax = newymin + h
        
        #check if label within new crop
        if not (newxmin > crop_shape[1] or newymin > crop_shape[0]  or newxmax < 0 or newymax < 0):
            
            if newxmin < 0:
                newxmin = 0
            if newymin < 0:
                newymin = 0
            
            if newxmax > crop_shape[1]:
                newxmax = crop_shape[1] -1
            if newymax > crop_shape[0]:
                newymax = crop_shape[0] -1

            new_w = newxmax - newxmin
            new_h = newymax - newymin
            
            #Only label if w and h of box greater than threshold.
            T = 5
            if new_w > T and new_h > T:
                new_data.append( [newxmin, newymin, new_w, new_h ] )
                
    return new_data
        


def get_split_im_data_from_full_im_data(key, data, crop_shape, im_shape, grid_shape ): #[[x,y,w,h], [x,y,w,h]]
    
    grid_h = crop_shape[0]/grid_shape[0]
    grid_w = crop_shape[1]/grid_shape[1]
    
    minxs, minys = get_minxys(*im_shape, *crop_shape)
    
    split_im_data={}
    
    for minx in minxs:
        for miny in minys:
            grid_minx = int(minx/grid_w)
            grid_miny = int(miny/grid_h)
            
            key_split_im = build_split_im_key(key, grid_minx, grid_miny )            
            new_bbox_coords = get_new_bbox_coords_from_split_im(data, minx, miny, crop_shape)
            split_im_data = {**split_im_data, key_split_im: new_bbox_coords}
            
    return split_im_data
            

            
            
def read_coco_annotations_test(file_path, crop_shape=(1200,1200), im_shape=(2400,3200), grid_shape=(3,3)):

    data = read_coco_annotations(file_path) # {key: [ [x,y,w,h], [x,y,w,h] ],
                                            #  key2:[[x,y,w,h]]}

    new_data = {}# {key_mingridy_mingridx: [ [x,y,w,h], [x,y,w,h] ],
                 #  key_mingridy_mingridx: [ [x,y,w,h], [x,y,w,h] ],
    
    for key in data.keys():
        split_im_data = get_split_im_data_from_full_im_data(key, data[key], crop_shape, im_shape, grid_shape  )
        new_data = {**new_data, **split_im_data}
    
    return new_data


#crop_shape=(1200,1200)
#im_shape=(2400,3200)
#grid_shape=(3,3)
#file_path = './data/data_external/annotations/val2020_simple.json'
#test_key = 'sep19_102MEDIA_DJI_0423.JPG'
#data = read_coco_annotations(file_path)


def get_image_crop(image, key, crop_shape=(1200,1200), im_shape=(2400,3200), grid_shape=(3,3)):
            
    #key format: fullimkey_gridminx_gridminy -> need to extract correct crop of image
    key_split = key.split('_')
    gridminx = int(key_split[-2])
    gridminy = int(key_split[-1])
    grid_h = crop_shape[0]/grid_shape[0]
    grid_w = crop_shape[1]/grid_shape[1]
    
    crop_minx = int( gridminx*grid_w )
    crop_maxx = int( (gridminx + grid_shape[1])*grid_w )
    crop_miny = int( gridminy*grid_h )
    crop_maxy = int( (gridminy + grid_shape[0])*grid_h )
    
    return image[ crop_miny:crop_maxy ,  crop_minx:crop_maxx , : ]


def show_im_with_grid_labels(im, grid, grid_shape, grid_gt=None):
    print('show_im_with_grid_labels',im.shape, grid.shape, grid_shape )
    fig,ax = plt.subplots(figsize=(10,10))
    grid_2d = grid.reshape(-1, grid_shape[1])
    if(grid_gt):
        grid_2d_gt = grid_gt.reshape(-1, grid_shape[1])
    
    print(grid_2d)
    grid_w = im.shape[1]/grid_2d.shape[1]
    grid_h = im.shape[0]/grid_2d.shape[0]
    
    ax.imshow(im[:,:,:3])
    
    #Draw Grid
    for x in range(grid_shape[1]):
        for y in range(grid_shape[0]):
            rect = patches.Rectangle((x*grid_w,y*grid_h),grid_w,grid_h,linewidth=2,edgecolor='white',facecolor='none')      
            ax.add_patch(rect)
    
    #Draw predictions/ground truth
    for x in range(grid_shape[1]):
        for y in range(grid_shape[0]):
            if grid_gt:
                if grid_2d_gt[y,x] > 0:
                    rect = patches.Rectangle((x*grid_w,y*grid_h),grid_w,grid_h,linewidth=5,edgecolor='green',facecolor='none')  
                    ax.add_patch(rect)
                
            if grid_2d[y,x] > 0:
                rect = patches.Rectangle((x*grid_w,y*grid_h),grid_w,grid_h,linewidth=4,edgecolor='red', ls='- ', facecolor='none')  
                ax.add_patch(rect)
    plt.draw()
    plt.show()
    
    
    
def read_coco_annotations(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        
    return data


def intersection_degree(bbox, box):
    xmin1, ymin1 = bbox[0]
    xmax1, ymax1 = bbox[1]
    xmin2, ymin2 = box[0]
    xmax2, ymax2 = box[1]
    
    A1 = (xmax1 - xmin1)*(ymax1-ymin1)
    A2 = (xmax2 - xmin2)*(ymax2-ymin2)
    
    dx_inter = min(xmax1,xmax2) - max(xmin1, xmin2)
    dy_inter = min(ymax1, ymax2) - max(ymin1, ymin2)
    A_inter=0
    if (dx_inter > 0) and (dy_inter > 0 ):
        A_inter = dx_inter*dy_inter
        
    return A_inter / A1


def grid_cell_has_label(grid_xmin, grid_ymin, grid_xmax, grid_ymax, labels):

    grid_geom = [ [grid_xmin, grid_ymin ], [grid_xmax, grid_ymax ] ]
    
    for minx, miny, w, h in labels:
        label_geom = [[ minx, miny ], [ minx + w, miny + w ]]
        
        if intersection_degree(label_geom, grid_geom) > 0.15:
            return True
        
    return False


def get_grid(bboxes, im_shape, grid_shape):

    grid_h = math.floor(im_shape[0]/ grid_shape[0])
    grid_w = math.floor(im_shape[1]/ grid_shape[1])
    
    grid = np.zeros(grid_shape)
    
    for x in range(grid_shape[1]):
        for y in range(grid_shape[0]):
            if grid_cell_has_label(x*grid_w, y*grid_h, (x+1)*grid_w,(y+1)*grid_h, bboxes ):
                grid[y,x] = 1
    
    return grid.flatten()


