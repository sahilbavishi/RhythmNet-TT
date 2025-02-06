import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import time
from storage_utils import save_statistics
from matplotlib import pyplot as plt

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

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=weight_decay_coefficient)
        self.learning_rate_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=0.00002)

        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.join(self.experiment_folder, "result_outputs")
        self.experiment_saved_models = os.path.join(self.experiment_folder, "saved_models")

        self.best_val_model_idx = 0
        self.best_val_model_acc = 0.0

        if not os.path.exists(self.experiment_folder):
            os.makedirs(self.experiment_logs)
            os.makedirs(self.experiment_saved_models)

        self.num_epochs = num_epochs
        #Two loss functions - one for position, one for class
        self.classifier_criterion = nn.CrossEntropyLoss().to(self.device)
        self.position_criterion=nn.CrossEntropyLoss().to(self.device)

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
        loss = self.classifier_criterion(output[:,:,:-1], y[:,:,:-1].type(torch.float32))+self.position_criterion(output[:,:,-1], y[:,:,-1].type(torch.float32)) #convert to float32 to do the measuring
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learning_rate_scheduler.step()

        #Accuracy is just the class accuracy at the moment - nothing to do with the position
        # _, predicted = torch.max(output[:,:,:-1], 2,keepdim=True)
        predicted_classes=torch.argmax(output[:,:,:-1],dim=-1)
        real_classes=torch.argmax(y[:,:,:-1],dim=-1)
        accuracy = (predicted_classes==real_classes).float().mean().item()
        return loss.item(), accuracy

    def run_evaluation_iter(self, x, y):
        self.eval()
        x, y = x.float().to(self.device), y.long().to(self.device)
        output = self.model(x)

        loss = self.classifier_criterion(output[:,:,:-1], y[:,:,:-1].type(torch.float32))+self.position_criterion(output[:,:,-1], y[:,:,-1].type(torch.float32)) #convert to float32 to do the measuring

        #Accuracy is just the class accuracy at the moment - nothing to do with the position

        predicted_classes=torch.argmax(output[:,:,:-1],dim=-1)
        real_classes=torch.argmax(y[:,:,:-1],dim=-1)
        print(predicted_classes)
        print(real_classes)
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
