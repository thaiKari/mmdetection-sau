#Adapted from starter code
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from dataloaders import load_sheep_grid_multiband, load_sheep_grid_data
from utils import to_cuda, compute_loss_and_accuracy, save_checkpoint
from resnet import ResNet, ResNetEnsembleInfraredRGB
from loss_functions import cross_entropy_cifar_loss, MultiLabelSoftMarginLossIgnoreEdgeCases
from pathlib import Path
from datetime import datetime
import time
import numpy as np


class Trainer:

    def __init__(self,
                 which_gpu,
                 img_type,
                 batch_size,
                 learning_rate,
                 start_from,
                 include_msx,
                 epochs,
                 early_stop_count,
                 train_layer2,
                 rgb_resize_shape,
                 infrared_resize_shape
                ):
        """
        Initialize our trainer class.
        Set hyperparameters, architecture, tracking variables etc.
        """
        # Define hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_count = early_stop_count
        self.img_type=img_type
        self.include_msx = include_msx
        self.checkpoint_path =   'Work_dirs/work_dirs_external/' + self.img_type + '/' + datetime.now().strftime("%Y%m%d_%H%M")
        self.which_gpu = which_gpu
        self.rgb_resize_shape = rgb_resize_shape
        self.infrared_resize_shape = infrared_resize_shape
        
        #Extra ensemble parameters
        if img_type == 'ensemble':
            time_stamp_rgb = '20200222_0826'
            state_file_rgb = '/model_best.pth.tar'
            model_path_rgb = './Work_dirs/work_dirs_external/rgb/' + time_stamp_rgb + state_file_rgb
            model_rgb = ResNet(image_channels=3, num_classes=9)
            model_rgb.load_state_dict(torch.load(model_path_rgb)['state_dict'])

            time_stamp_infrared = '20200221_1432'
            state_file_infrared = '/model_best.pth.tar'
            model_path_infrared = './Work_dirs/work_dirs_external/infrared/' + time_stamp_infrared  + state_file_infrared
            model_infrared = ResNet(image_channels=3, num_classes=9)
            model_infrared.load_state_dict(torch.load(model_path_infrared)['state_dict'])
        
        
        
        # Make log folder and files
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.checkpoint_path,"TRAIN_LOSS.txt"), "w") as file:
            file.write("")

        with open(os.path.join(self.checkpoint_path,"VALIDATION_LOSS.txt"), "w") as file:
            file.write("")
            
        with open(os.path.join(self.checkpoint_path,"META.txt"), "w") as file:
            s = "num_epochs = " + str(self.epochs) + "\n" 
            s = s + "batch_size  = " + str(self.batch_size) +"\n" 
            s = s + "learning_rate  = " + str(self.learning_rate) +"\n" 
            s = s + "early_stop_count  = " + str(self.early_stop_count) +"\n"                          
            s = s + "img_type  = " + str(self.img_type) +"\n"   
            s = s + "include_msx  = " + str(self.include_msx) +"\n" 
            s = s + "train_layer2  = " + str(train_layer2) +"\n" 
            s = s + "which_gpu  = " + str(which_gpu) + "\n"
            s = s + "rgb_resize_shape  = " + str(rgb_resize_shape) + "\n"
            s = s + "infrared_resize_shape  = " + str(infrared_resize_shape) + "\n"
                      
            
            if self.img_type == 'ensemble':
                s = s + "time_stamp_rgb = " + str(time_stamp_rgb) + "\n"                       
                s = s + "time_stamp_infrared = " + str(time_stamp_infrared) + "\n" 
                          

            if start_from:
                s = s + "start_from = " + str(start_from) + "\n" 
                
            file.write(s)
        

        self.loss_criterion = MultiLabelSoftMarginLossIgnoreEdgeCases      
        
        
        # Initialize the model
        if self.img_type == 'ensemble':
            self.model = ResNetEnsembleInfraredRGB(num_classes = 9,
                                                   ResNetRGB = model_rgb,
                                                   ResNetIR = model_infrared,
                                                   train_layer2=train_layer2,
                                                   rgb_size = self.rgb_resize_shape[0],
                                                   infrared_size = self.infrared_resize_shape[0]                                           
                                                  ) 
        else:            
            self.model = ResNet(image_channels=3, num_classes=9, train_layer2=train_layer2)  
        
        # Transfer model to GPU VRAM, if possible.
        self.model = to_cuda(self.model, self.which_gpu)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         self.learning_rate)
        
        if start_from:
            self.model.load_state_dict(torch.load(start_from)['state_dict'])
            self.optimizer.load_state_dict(torch.load(start_from)['optimizer_state_dict'])
            
        
        
        # Load our dataset
        labels_val_path = 'annotations/val2020_simple.json'
        labels_train_path = 'annotations/train2020_simple.json'
        
        if self.include_msx:
            labels_val_path = 'annotations/val_labels_infrared_and_msx_simple.json'
            labels_train_path =  'annotations/train_labels_infrared_and_msx_simple.json'
            
        if self.img_type == 'ensemble':
            self.dataloader_train, self.dataloader_val = load_sheep_grid_multiband(self.batch_size,
                                                                              rgb_resize_shape=self.rgb_resize_shape,
                                                                              infrared_resize_shape=self.infrared_resize_shape)
        else:
            self.dataloader_train, self.dataloader_val = load_sheep_grid_data(self.batch_size,
                                                                          img_type=self.img_type,
                                                                          include_msx=self.include_msx,
                                                                         labels_val_path=labels_val_path,
                                                                         labels_train_path=labels_train_path,                                                                              rgb_resize_shape=self.rgb_resize_shape,                                                                      infrared_resize_shape=self.infrared_resize_shape
                                                                             ) 

        self.validation_check = len(self.dataloader_train) // 2

        # Tracking variables
        self.VALIDATION_LOSS = []
        self.TRAIN_LOSS = []
        self.TRAIN_ACC = []
        self.VALIDATION_ACC = []


    def validation_epoch(self, epoch):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()

        # Compute for training set
        train_loss = compute_loss_and_accuracy(
            self.dataloader_train, self.model, self.loss_criterion, which_gpu=self.which_gpu, ensemble_learning= self.img_type == 'ensemble'
        )
        #self.TRAIN_ACC.append(train_acc)
        self.TRAIN_LOSS.append(train_loss)

        # Compute for validation set
        validation_loss= compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion, which_gpu=self.which_gpu, ensemble_learning= self.img_type == 'ensemble'
        )
        #self.VALIDATION_ACC.append(validation_acc)
        
        self.VALIDATION_LOSS.append(validation_loss)
        is_best = validation_loss <= min(self.VALIDATION_LOSS)
        print("Current validation loss:", validation_loss, " train_loss:", train_loss)
        
        
        
        if epoch >= 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, is_best, checkpoint_path = self.checkpoint_path, filename=os.path.join(self.checkpoint_path, 'epoch_'+str(epoch + 1) +'.pth.tar') )
            
        with open(os.path.join(self.checkpoint_path,"TRAIN_LOSS.txt"), "a") as file:
            file.write(str(train_loss) + ' ')
        with open(os.path.join(self.checkpoint_path,"VALIDATION_LOSS.txt"), "a") as file:
            file.write(str(validation_loss) + ' ')


            


        self.model.train()

    def should_early_stop(self):
        """
        Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        if len(self.VALIDATION_LOSS) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = self.VALIDATION_LOSS[-self.early_stop_count:]
        previous_loss = relevant_loss[0]
        for current_loss in relevant_loss[1:]:
            # If the next loss decrease, early stopping criteria is not met.
            if current_loss < previous_loss:
                return False
            previous_loss = current_loss
        return True

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        start = time.time()
        # Track initial loss/accuracy
        self.validation_epoch(-1)
        for epoch in range(self.epochs):
            print('Epoch [{}/{}]'.format(epoch, self.epochs))
            # Perform a full pass through all the training samples
            for batch_it, sample in enumerate(self.dataloader_train):
                if self.img_type == 'ensemble':
                    X_batch_rgb = sample['rgb']
                    X_batch_infrared = sample['infrared']
                    Y_batch = sample['grid']

                    X_batch_rgb = to_cuda(X_batch_rgb, self.which_gpu)
                    X_batch_infrared = to_cuda(X_batch_infrared, self.which_gpu)
                    Y_batch = to_cuda(Y_batch, self.which_gpu)

                    # Perform the forward pass
                    predictions = self.model(X_batch_rgb, X_batch_infrared)
                else:
                    X_batch = sample['image']
                    Y_batch = sample['grid']

                    X_batch = to_cuda(X_batch, self.which_gpu)
                    Y_batch = to_cuda(Y_batch, self.which_gpu)

                    predictions = self.model(X_batch)
                
                # Compute the cross entropy loss for the batch
                loss = self.loss_criterion(predictions, Y_batch)
                #print('loss', loss)

                # Backpropagation
                loss.backward()

                # Gradient descent step
                self.optimizer.step()
                
                # Reset all computed gradients to 0
                self.optimizer.zero_grad()
                 # Compute loss/accuracy for all three datasets.
                if batch_it % self.validation_check == 0:
                    self.validation_epoch(epoch)
                    # Check early stopping criteria.
                    if self.should_early_stop():
                        print("Early stopping.")
                        end = time.time()
        
                        with open(os.path.join(self.checkpoint_path,"META.txt"), "a") as file:
                            file.write( "start = " + str(start) + "\n" +
                                          "end  = " + str(end) +"\n" + 
                                        "time seconds = " + str(end - start) +"\n" +
                                         "early_stopped = " + 'True' +"\n" +
                                         'best_val_loss= ' + str(np.min(self.VALIDATION_LOSS)) +"\n" +
                                         'best_train_loss= ' + str(np.min(self.TRAIN_LOSS)) +"\n"
                                        )
                        return
                    
        end = time.time()
        
        with open(os.path.join(self.checkpoint_path,"META.txt"), "a") as file:
            file.write( "start = " + str(start) + "\n" +
                          "end  = " + str(end) +"\n" + 
                        "time seconds = " + str(end - start) +"\n" +
                         "early_stopped = " + 'False' +"\n"+
                         'best_val_loss= ' + str(np.min(self.VALIDATION_LOSS)) +"\n" +
                         'best_train_loss= ' + str(np.min(self.TRAIN_LOSS)) +"\n"
                      )
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=0, type=int, help="Which gpu to run training on")
    parser.add_argument("--img_type", default='rgb', type=str, help="rgb or infrared")
    parser.add_argument("--batch_size", default=12, type=int, help="number of images in a batch")
    parser.add_argument("--learning_rate", default='5e-5', type=str, help="the learning rate")
    parser.add_argument("--start_from", default=None, type=str, help="load weights from checkpoint at this file location")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs to train")
    parser.add_argument("--early_stop_count", default=8, type=int, help="stop training if no improvement after n epochs") 
    parser.add_argument("--include_msx", default=False, type=str2bool, help="should training include msx images")
    parser.add_argument("--train_layer2", default=True, type=str2bool, help="should also unfreeze layer2 of ResNet")
    parser.add_argument("--rgb_resize_shape", default=1280, type=int, help="size to resize rgb crop to") 
    parser.add_argument("--infrared_resize_shape", default=160, type=int, help="size to resize infrared crop to") 
    args = parser.parse_args()


    trainer = Trainer(which_gpu=args.gpu,
                      img_type=args.img_type,
                      batch_size= args.batch_size,
                      learning_rate= float(args.learning_rate),
                      start_from=args.start_from,
                      include_msx=args.include_msx,
                      epochs=args.epochs,
                      early_stop_count = args.early_stop_count,
                      train_layer2 = args.train_layer2,
                      rgb_resize_shape=(args.rgb_resize_shape, args.rgb_resize_shape),
                      infrared_resize_shape=(args.infrared_resize_shape, args.infrared_resize_shape)
                     )
    

    trainer.train() 
    


    print("Best validation loss:", min(trainer.VALIDATION_LOSS))