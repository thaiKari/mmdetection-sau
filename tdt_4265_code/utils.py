#Starter code from https://raw.githubusercontent.com/hukkelas/TDT4265-A3-starter-code/master/utils.py
import torch
import shutil
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os

def to_cuda(elements):
    """
    Transfers elements to GPU memory, if a nvidia- GPU is available.
    Args:
        elements: A list or a single pytorch module.
    Returns:
        The same list transferred to GPU memory
    """

    if torch.cuda.is_available(): # Checks if a GPU is available for pytorch
        if isinstance(elements, (list, tuple)):
            return [x.cuda() for x in elements] # Transfer each index of the list to GPU memory
        return elements.cuda()
    return elements


def compute_loss_and_accuracy(dataloader, model, loss_criterion):
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

    for (X_batch, Y_batch, _) in dataloader:
        #print(X_batch.size(), Y_batch.size())
        
        # Transfer images/labels to GPU VRAM, if possible
        X_batch = to_cuda(X_batch)
        Y_batch = to_cuda(Y_batch)
        # Forward pass the images through our model
        output_probs = model(X_batch) 
        #Size output_probs: [batch_size, n_classes]
        #Size output_probs: [batch_size] (the class number for each image in the batch)
        # Compute loss
        
        #print('output_probs', output_probs.size())
        loss = loss_criterion(output_probs, Y_batch)
        print('loss', loss)

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
    torch.save(state, filename)
    print('saving state epoch', state['epoch'])
    if is_best:
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