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
from collections import defaultdict #For tracking TP,FP,TN
from performance_metrics import get_TP_TN_FP_FN, get_metrics, get_windows_TP_FP_FN, get_windows_metrics
from loss_functions import FocalLoss, CreateHeartbeatWindow, GboxIoULoss
            

class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, val_data,
                 test_data, weight_decay_coefficient, learning_rate, use_gpu, continue_from_epoch=-1):
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay_coefficient)
        
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
        self.position_criterion2 = GboxIoULoss(device=self.device,maxtimeframe=1).to(self.device)
        self.WindowMaker = CreateHeartbeatWindow(maxtimeframe=1,sigma=0.4,device=self.device).to(self.device)

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
        x, y = x.float().to(self.device), y.float().to(self.device)
        output = self.model(x)
        #Get predicted and real classes
        predicted_classes=torch.argmax(output[:,:,:-2],dim=-1) # [batch, num_pred]
        real_classes=torch.argmax(y[:,:,:-1],dim=-1) # [batch, num_gt]
        
        predicted_positions=torch.sort(output[:,:,-2:],dim=2)[0] #Sort the predicted boxes, [batch, num_pred, 2]
        TargetWindows=self.WindowMaker(y[:,:,-1].type(torch.float32)) #window of heartbeat, [batch, num_gt, 2]

        #print("predicted_positions", predicted_positions[0,:,:])
        #print("TargetWindows", TargetWindows[0,:,:])


        #Change the output of the positions to make sure that the model does not need to worry about positions when it has classified a null beat
        output[:,:,-1][predicted_classes==4]=1 #Set positions to max when the model has predicted the null class
        
        #Loss calculations
        ClassLoss=self.classifier_criterion(output[:,:,:-2], y[:,:,:-1].type(torch.float32))
        PosLoss=self.position_criterion1(predicted_positions, TargetWindows)+self.position_criterion2(predicted_positions,TargetWindows)
        loss = ClassLoss+PosLoss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learning_rate_scheduler.step()

        #Metrics calculations
        TP, TN, FP, FN = get_TP_TN_FP_FN(predicted_classes, real_classes, num_classes=5,device=self.device)
        accuracy, precision, recall, f1 = get_metrics(TP, TN, FP, FN)

        TP, FP, FN = get_windows_TP_FP_FN(predicted_positions.detach(), TargetWindows.detach(), predicted_classes, real_classes,device=self.device)
        precision_window, recall_window, f1_window = get_windows_metrics(TP, FP, FN)
    
        return loss.item(), accuracy, precision, recall, f1, precision_window, recall_window, f1_window

    def run_evaluation_iter(self, x, y):
        self.eval()
        x, y = x.float().to(self.device), y.float().to(self.device)
        output = self.model(x)

        # Get predictions and ground truths
        predicted_classes=torch.argmax(output[:,:,:-2],dim=-1) # [batch, num_pred]
        real_classes=torch.argmax(y[:,:,:-1],dim=-1) # [batch, num_gt]
        
        predicted_positions=torch.sort(output[:,:,-2:],dim=2)[0] #Sort the predicted boxes, [batch, num_pred, 2]
        TargetWindows=self.WindowMaker(y[:,:,-1].type(torch.float32)) # window of heartbeat, [batch, num_gt, 2]

        #Change the output of the positions to make sure that the model does not need to worry about positions when it has classified a null beat
        output[:,:,-1][predicted_classes==4]=1 #Set positions to max when the model has predicted the null class

        #Loss calculations
        ClassLoss=self.classifier_criterion(output[:,:,:-2], y[:,:,:-1].type(torch.float32))
        PosLoss=self.position_criterion1(predicted_positions, TargetWindows)+self.position_criterion2(predicted_positions,TargetWindows)
        loss = ClassLoss+PosLoss
        
        #Metrics calculations, just for class at the moment - nothing to do with the position
        TP, TN, FP, FN = get_TP_TN_FP_FN(predicted_classes, real_classes, num_classes=5,device=self.device)
        accuracy, precision, recall, f1 = get_metrics(TP, TN, FP, FN)

        TP, FP, FN = get_windows_TP_FP_FN(predicted_positions.detach(), TargetWindows.detach(), predicted_classes, real_classes,device=self.device)
        precision_window, recall_window, f1_window = get_windows_metrics(TP, FP, FN)

        #print("predicted_positions", predicted_positions[0,:,:])
        #print("TargetWindows", TargetWindows[0,:,:])
    
        return loss.item(), accuracy, precision, recall, f1, precision_window, recall_window, f1_window

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
        state = torch.load(model_path,weights_only=False)
        self.model.load_state_dict(state['network'])
        self.optimizer.load_state_dict(state['optimizer'])
        return state, state['best_val_model_idx'], state['best_val_model_acc']

    def run_experiment(self):
        total_losses = {"train_acc": [], "train_loss": [], "train_precision": [], "train_recall": [], "train_f1": [],
                        "train_precision_window": [], "train_recall_window": [], "train_f1_window": [],
                        "val_acc": [], "val_loss": [], "val_precision": [], "val_recall": [], "val_f1": [],
                        "val_precision_window": [], "val_recall_window": [], "val_f1_window": []}

        for epoch in range(self.starting_epoch, self.num_epochs):
            epoch_start_time = time.time()
            current_epoch_losses = {"train_acc": [], "train_loss": [], "train_precision": [], "train_recall": [], "train_f1": [],
                                    "train_precision_window": [], "train_recall_window": [], "train_f1_window": [],
                                    "val_acc": [], "val_loss": [], "val_precision": [], "val_recall": [], "val_f1": [],
                                    "val_precision_window": [], "val_recall_window": [], "val_f1_window": []}

            self.train()
            with tqdm.tqdm(total=len(self.train_data)) as pbar:
                for x, y in self.train_data:
                    loss, acc, prec, rec, f1, prec_window, rec_window, f1_window = self.run_train_iter(x, y)
                    current_epoch_losses['train_loss'].append(loss)
                    current_epoch_losses['train_acc'].append(acc['micro_avg_accuracy'])
                    current_epoch_losses['train_precision'].append(prec['macro_avg_precision'])
                    current_epoch_losses['train_recall'].append(rec['macro_avg_recall'])
                    current_epoch_losses['train_f1'].append(f1['macro_avg_f1'])
                    current_epoch_losses['train_precision_window'].append(prec_window)
                    current_epoch_losses['train_recall_window'].append(rec_window)
                    current_epoch_losses['train_f1_window'].append(f1_window)
                    

                    pbar.update(1)
                    pbar.set_description(f"Epoch {epoch} - Loss: {loss:.4f}, Acc: {acc['micro_avg_accuracy']:.4f}, F1: {f1['macro_avg_f1']:.4f}")

            self.eval()
            with torch.no_grad():
                with tqdm.tqdm(total=len(self.val_data)) as pbar:
                    for x, y in self.val_data:
                        loss, acc, prec, rec, f1, prec_window, rec_window, f1_window = self.run_evaluation_iter(x, y)
                        current_epoch_losses['val_loss'].append(loss)
                        current_epoch_losses['val_acc'].append(acc['micro_avg_accuracy'])
                        current_epoch_losses['val_precision'].append(prec['macro_avg_precision'])
                        current_epoch_losses['val_recall'].append(rec['macro_avg_recall'])
                        current_epoch_losses['val_f1'].append(f1['macro_avg_f1'])
                        current_epoch_losses['val_precision_window'].append(prec_window.cpu())
                        current_epoch_losses['val_recall_window'].append(rec_window.cpu())
                        current_epoch_losses['val_f1_window'].append(f1_window.cpu())

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

        test_losses = {"test_acc": [], "test_loss": [], "test_precision": [], "test_recall": [], "test_f1": [],
                       "test_precision_window": [], "test_recall_window": [], "test_f1_window": []}
        with tqdm.tqdm(total=len(self.test_data)) as pbar:
            for x, y in self.test_data:
                loss, acc, prec, rec, f1, 
                test_losses['test_loss'].append(loss)
                test_losses['test_acc'].append(acc['micro_avg_accuracy'])
                test_losses['test_precision'].append(prec['macro_avg_precision'])
                test_losses['test_recall'].append(rec['macro_avg_recall'])
                test_losses['test_f1'].append(f1['macro_avg_f1'])
                test_losses['test_precision_window'].append(prec_window)
                test_losses['test_recall_window'].append(rec_window)
                test_losses['test_f1_window'].append(f1_window)
                pbar.update(1)

        save_statistics(self.experiment_logs, 'test_summary.csv', test_losses, 0, continue_from_mode=False)
        return total_losses, test_losses
