#Adapted from starter code
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from dataloaders import load_cifar10, load_sheep_grid_data
from utils import to_cuda, compute_loss_and_accuracy, save_checkpoint
from resnet import ResNet
from loss_functions import cross_entropy_cifar_loss
from pathlib import Path
from datetime import datetime


class Trainer:

    def __init__(self):
        """
        Initialize our trainer class.
        Set hyperparameters, architecture, tracking variables etc.
        """
        # Define hyperparameters
        self.epochs = 20
        self.batch_size = 3
        self.learning_rate = 5e-5
        self.early_stop_count = 8
        self.visOnly = False
        self.checkpoint_path =  'Work_dirs/work_dirs_external/' + datetime.now().strftime("%Y%m%d_%H%M")
        
        # Make log folder and files
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.checkpoint_path,"TRAIN_LOSS.txt"), "w") as file:
            file.write("")

        with open(os.path.join(self.checkpoint_path,"VALIDATION_LOSS.txt"), "w") as file:
            file.write("")

        


        # Since we are doing multi-class classification, we use the CrossEntropyLoss
        self.loss_criterion = nn.MultiLabelSoftMarginLoss()#cross_entropy_cifar_loss # # # nn.CrossEntropyLoss()
        # Initialize the mode
        self.model = ResNet(image_channels=4, num_classes=9)
        # Transfer model to GPU VRAM, if possible.
        self.model = to_cuda(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        #self.optimizer = torch.optim.SGD(self.model.parameters(),
        #                                 self.learning_rate)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         self.learning_rate)
        
        # Load our dataset
        self.dataloader_train, self.dataloader_val = load_sheep_grid_data(self.batch_size, visOnly = self.visOnly) #load_cifar10(self.batch_size) #

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
            self.dataloader_train, self.model, self.loss_criterion
        )
        #self.TRAIN_ACC.append(train_acc)
        self.TRAIN_LOSS.append(train_loss)

        # Compute for validation set
        validation_loss= compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion
        )
        #self.VALIDATION_ACC.append(validation_acc)
        
        self.VALIDATION_LOSS.append(validation_loss)
        is_best = validation_loss <= min(self.VALIDATION_LOSS)
        print("Current validation loss:", validation_loss, " train_loss:", train_loss)
        
        
        
        if epoch >= 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
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
        # Track initial loss/accuracy
        self.validation_epoch(-1)
        for epoch in range(self.epochs):
            print('Epoch [{}/{}]'.format(epoch, self.epochs))
            # Perform a full pass through all the training samples
            for batch_it, (X_batch, Y_batch, _) in enumerate(self.dataloader_train):
                # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32]
                # Y_batch is the CIFAR10 image label. Shape: [batch_size]
                # Transfer images / labels to GPU VRAM, if possible
                X_batch = to_cuda(X_batch)
                Y_batch = to_cuda(Y_batch)

                # Perform the forward pass
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
                        return


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()

    os.makedirs("plots", exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(12, 8))
    plt.title("Cross Entropy Loss")
    plt.plot(trainer.VALIDATION_LOSS, label="Validation loss")
    plt.plot(trainer.TRAIN_LOSS, label="Training loss")

    plt.legend()
    plt.savefig(os.path.join("plots", "final_loss.png"))
    plt.show()

    #plt.figure(figsize=(12, 8))
    #plt.title("Accuracy")
    #plt.plot(trainer.VALIDATION_ACC, label="Validation Accuracy")
    #plt.plot(trainer.TRAIN_ACC, label="Training Accuracy")

    #plt.legend()
    #plt.savefig(os.path.join("plots", "final_accuracy.png"))
    #plt.show()


    print("Final validation loss:", trainer.VALIDATION_LOSS[-trainer.early_stop_count])