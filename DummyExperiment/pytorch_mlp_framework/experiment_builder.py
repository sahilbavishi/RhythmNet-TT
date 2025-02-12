import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.ops as ops
import tqdm
import os
import numpy as np
import time
from storage_utils import save_statistics
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

class FocalLoss(nn.Module):
    """
    A class for calulcating the focal loss for the loss function
    """
    def __init__(self, Theta,Gamma):
        """
        Initialise the focal loss class

        Inputs
        ------
        Theta: The theta parameter

        Gamma: The gamma parameter
        """
        super(FocalLoss,self).__init__()
        self.theta=Theta
        self.gamma=Gamma
    def __call__(self,Preds,RealVals):
        """
        Inputs
        ------
        Preds: The predicted probabilities of the classes

        RealVals: One-hot encodings of the correct classes
        """
        #First, we are going to get the correct classes from the realvals
        real_classes=torch.argmax(RealVals,dim=-1)
        #Get the probabilities predicted of the targets
        Ps=torch.gather(Preds,dim=2,index=torch.unsqueeze(real_classes,2))
        #Calc loss:
        Loss=-self.theta*(1-Ps)**2*torch.log(Ps)

        return torch.sum(Loss) #Return the sum of the losses

class CreateHeartbeatWindow(nn.Module):
    def __init__(self, maxtimeframe,sigma):
        super(CreateHeartbeatWindow,self).__init__()
        self.maxtimeframe=maxtimeframe
        self.sigma=sigma
    def __call__(self, InputPositions):
        """
        Function to return the window of the heartbeat

        InputPositons size: [Batchsize, NumberofClassifications, 1] (The positions are at the end)
        """
        #To do this, I am first going to change any 0s to above the maximum length, as they are at the end
        InputPositions[InputPositions==0]=self.maxtimeframe*2
        #Append one more as well
        InputPositions=torch.cat((torch.unsqueeze(InputPositions,2),self.maxtimeframe*2*torch.ones([InputPositions.shape[0],1,1])),dim=1)
        #Now get the start positions and the end positions of the windows:
        StartEnd=InputPositions[:,:-1,:]*self.sigma+(1-self.sigma)*InputPositions[:,1:,:]
        #Append a 0 to the start
        StartEnd=torch.cat((torch.zeros([StartEnd.shape[0],1,1]),StartEnd),dim=1)
        #Make sure we are not giving too wide a window
        StartEnd[StartEnd>self.maxtimeframe]=self.maxtimeframe
        #Return the start and end position by concating them on the 2nd dimension
        StartEnd=torch.cat((StartEnd[:,:-1,:],StartEnd[:,1:,:]),dim=2)
        return StartEnd


class GboxIoULoss(nn.Module):
    """
    A class that does the box gIoU loss function, but also deals with the range for a heart beat to be in.
    This is based off the paper we are doing our base model on
    """
    def __init__(self,delta=float(10e-7),maxtimeframe=1080):
        """
        Initialise the gboxiou loss module

        Inputs
        ------

        delta: the delta parameter for the actual loss function

        maxtimeframe: the number of discrete recordings in the time window chosen (1080 is 3 seconds)
        """
        super(GboxIoULoss,self).__init__()
        self.delta=delta
        self.maxtimeframe=maxtimeframe
    def __call__(self, Preds,TargetWindows):
        """
        Calculate the loss, by creating the windows and then using the loss function

        Inputs
        ------
        Preds: The predicted windows of the heartbeats

        Targets: The actual windows of the heartbeats
        """

        #Get the start of window and end of window
        out1,out2=Preds[:,:,0],Preds[:,:,1]
        tgt1,tgt2=TargetWindows[:,:,0],TargetWindows[:,:,1]
        #Calculate intersetctions,unions and areas
        Intersection=torch.max(torch.zeros_like(out1),torch.min(out2,tgt2)-torch.max(out1,tgt1))
        Union=tgt2-tgt1+out2-out1-Intersection #Union is the size of both minus the intersection between them
        Area=torch.max(tgt2,out2)-torch.min(tgt1,out1) #This is the shortest area that encompases both windows

        #Calc loss
        Loss=1-(Intersection/Union-(Area-Union)/(Area+self.delta))

        return torch.sum(Loss) #Return the sum of all the losses 


        


        
        




class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, val_data,
                 test_data, weight_decay_coefficient, use_gpu, continue_from_epoch=-1):
        super(ExperimentBuilder, self).__init__()

        self.experiment_name = experiment_name
        self.model = network_model

        if torch.cuda.device_count() >= 1 and use_gpu:
            self.device = torch.device('cuda')
            self.model.to(self.device)  # Send the model to the GPU
            print('Using GPU:', self.device)
        else:
            self.device = torch.device('cpu')
            print('Using CPU:', self.device)

        self.model.reset_parameters()  # Re-initialize network parameters
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        '''old learning rate scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=weight_decay_coefficient)
        self.learning_rate_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=0.00002)
        '''
        #new learning rate scheduler
        # Define the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=weight_decay_coefficient)
        
        # Warm-up scheduler (first 10 epochs)
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01, total_iters=10)
        
        # Cosine annealing scheduler (after warm-up)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs - 10, eta_min=0.00002)
        
        # Combine warm-up and cosine annealing
        self.learning_rate_scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[10])


        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.join(self.experiment_folder, "result_outputs")
        self.experiment_saved_models = os.path.join(self.experiment_folder, "saved_models")

        self.best_val_model_idx = 0
        self.best_val_model_acc = 0.0

        if not os.path.exists(self.experiment_folder):
            os.makedirs(self.experiment_logs)
            os.makedirs(self.experiment_saved_models)

        self.num_epochs = num_epochs
        #Three loss functions - two for position, one for class but the second position loss in the actual loss calc
        self.classifier_criterion = FocalLoss(Theta=1,Gamma=2).to(self.device) #Thetas may need to be tuned for what we want
        self.position_criterion1 = nn.L1Loss().to(self.device)
        self.position_criterion2 = GboxIoULoss(maxtimeframe=1080).to(self.device)
        self.WindowMaker = CreateHeartbeatWindow(maxtimeframe=1080,sigma=0.4).to(self.device)

        if continue_from_epoch >= 0:
            self.state, self.best_val_model_idx, self.best_val_model_acc = self.load_model(
                self.experiment_saved_models, "train_model", continue_from_epoch
            )
            self.starting_epoch = continue_from_epoch
        else:
            self.state = {}
            self.starting_epoch = 0

    def run_train_iter(self, x, y):
        self.train()
        x, y = x.float().to(self.device), y.long().to(self.device)
        output = self.model(x)
        #Get predicted and real classes
        predicted_classes=torch.argmax(output[:,:,:-2],dim=-1)
        real_classes=torch.argmax(y[:,:,:-1],dim=-1)
        
        predicted_positions=torch.sort(output[:,:,-2:],dim=2)[0] #Sort the predicted boxes

        #Change the output of the positions to make sure that the model does not need to worry about positions when it has classified a null beat
        output[:,:,-1][predicted_classes==5]=0 #Set positions to 0 when the model has predicted the null class
        
        TargetWindows=self.WindowMaker(y[:,:,-1].type(torch.float32))#Create window of heartbeat
        
        #Loss calculations
        ClassLoss=self.classifier_criterion(output[:,:,:-2], y[:,:,:-1].type(torch.float32))
        PosLoss=self.position_criterion1(predicted_positions, TargetWindows)+self.position_criterion2(predicted_positions,TargetWindows)
        loss = ClassLoss+PosLoss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learning_rate_scheduler.step()

        #Accuracy is just the class accuracy at the moment - nothing to do with the position
        accuracy = (predicted_classes==real_classes).float().mean().item()
        return loss.item(), accuracy

    def run_evaluation_iter(self, x, y):
        self.eval()
        x, y = x.float().to(self.device), y.long().to(self.device)
        output = self.model(x)
        #Get predicted and real classes
        predicted_classes=torch.argmax(output[:,:,:-2],dim=-1)
        real_classes=torch.argmax(y[:,:,:-1],dim=-1)
        
        predicted_positions=torch.sort(output[:,:,-2:],dim=2)[0] #Sort the predicted boxes

        #Change the output of the positions to make sure that the model does not need to worry about positions when it has classified a null beat
        output[:,:,-1][predicted_classes==5]=0 #Set positions to 0 when the model has predicted the null class
        
        TargetWindows=self.WindowMaker(y[:,:,-1].type(torch.float32))#Create window of heartbeat
        #Loss calculations
        ClassLoss=self.classifier_criterion(output[:,:,:-2], y[:,:,:-1].type(torch.float32))
        PosLoss=self.position_criterion1(predicted_positions, TargetWindows)+self.position_criterion2(predicted_positions,TargetWindows)
        loss = ClassLoss+PosLoss

        #Accuracy is just the class accuracy at the moment - nothing to do with the position
        accuracy = (predicted_classes==real_classes).float().mean().item()
        return loss.item(), accuracy

    def save_model(self, model_dir, model_name, epoch, best_val_model_idx, best_val_model_acc):
        self.state['network'] = self.model.state_dict()
        self.state['optimizer'] = self.optimizer.state_dict()
        self.state['epoch'] = epoch
        self.state['best_val_model_idx'] = best_val_model_idx
        self.state['best_val_model_acc'] = best_val_model_acc

        model_path = os.path.join(model_dir, f"{model_name}_{epoch}.pth")
        torch.save(self.state, model_path)

    def load_model(self, model_dir, model_name, epoch):
        model_path = os.path.join(model_dir, f"{model_name}_{epoch}.pth")
        state = torch.load(model_path)
        self.model.load_state_dict(state['network'])
        self.optimizer.load_state_dict(state['optimizer'])
        return state, state['best_val_model_idx'], state['best_val_model_acc']

    def run_experiment(self):
        total_losses = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}

        for epoch in range(self.starting_epoch, self.num_epochs):
            epoch_start_time = time.time()
            current_epoch_losses = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}

            self.train()
            with tqdm.tqdm(total=len(self.train_data)) as pbar:
                for x, y in self.train_data:
                    loss, acc = self.run_train_iter(x, y)
                    current_epoch_losses['train_loss'].append(loss)
                    current_epoch_losses['train_acc'].append(acc)
                    pbar.update(1)
                    pbar.set_description(f"Epoch {epoch} - Loss: {loss:.4f}, Acc: {acc:.4f}")

            self.eval()
            with torch.no_grad():
                with tqdm.tqdm(total=len(self.val_data)) as pbar:
                    for x, y in self.val_data:
                        loss, acc = self.run_evaluation_iter(x, y)
                        current_epoch_losses['val_loss'].append(loss)
                        current_epoch_losses['val_acc'].append(acc)
                        pbar.update(1)

            val_mean_acc = np.mean(current_epoch_losses['val_acc'])
            if val_mean_acc > self.best_val_model_acc:
                self.best_val_model_acc = val_mean_acc
                self.best_val_model_idx = epoch

            for key in current_epoch_losses:
                total_losses[key].append(np.mean(current_epoch_losses[key]))

            save_statistics(self.experiment_logs, 'summary.csv', total_losses, epoch, continue_from_mode=(epoch > 0))

            print(f"Epoch {epoch} completed in {time.time() - epoch_start_time:.2f}s. Best Val Acc: {self.best_val_model_acc:.4f}")
            self.save_model(self.experiment_saved_models, "train_model", epoch, self.best_val_model_idx, self.best_val_model_acc)

        print("Evaluating on test set...")
        self.load_model(self.experiment_saved_models, "train_model", self.best_val_model_idx)

        test_losses = {"test_acc": [], "test_loss": []}
        with torch.no_grad():
            with tqdm.tqdm(total=len(self.test_data)) as pbar:
                for x, y in self.test_data:
                    loss, acc = self.run_evaluation_iter(x, y)
                    test_losses['test_loss'].append(loss)
                    test_losses['test_acc'].append(acc)
                    pbar.update(1)

        save_statistics(self.experiment_logs, 'test_summary.csv', test_losses, 0, continue_from_mode=False)
        return total_losses, test_losses
