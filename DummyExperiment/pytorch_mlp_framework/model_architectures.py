import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_features, hidden_units, output_classes):
        """
        Initializes a fully connected network for tabular data.
        :param input_features: Number of input features (columns in the dataset).
        :param hidden_units: Number of units in the hidden layer.
        :param output_classes: Number of output classes (e.g., N and A).
        """
        super(FullyConnectedNetwork, self).__init__()
        self.fc1 = nn.Linear(input_features, hidden_units)  # First layer
        self.fc2 = nn.Linear(hidden_units, output_classes)  # Output layer
        self.relu = nn.ReLU()  # Activation function
        self.softmax = nn.Softmax(dim=1)  # For multi-class classification

    def forward(self, x):
        """
        Forward pass through the network.
        :param x: Input tensor.
        :return: Predicted logits.
        """
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

    def reset_parameters(self):
        """
        Resets the network parameters for reinitialization.
        """
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

class Bottleneck:
    def __init__(self, input_shape, k, c, n, s):
        self.input_shape = input_shape
        self.k = k
        self.c = c
        self.n = n
        self.s = s
        print(self.build_module())

    def build_module(self):

        self.layer_dict = nn.ModuleDict()

        x = torch.zeros((self.input_shape))
        out = x
        print("\n bottleneck 1 input shape: ",out.shape)
        self.layer_dict['fist_bottleneck_conv1d'] = nn.Conv1d(in_channels = out.shape[1], out_channels = self.c, kernel_size = 1, stride = self.s, padding = 0)
        self.layer_dict['bn0'] = nn.BatchNorm1d(num_features= self.c)
        
        out = self.layer_dict['fist_bottleneck_conv1d'].forward(out)
        out = self.layer_dict['bn0'].forward(out)
        out = F.relu(out)
        print("\n bottleneck 1st layer ouput shape: ",out.shape)

        self.layer_dict['depthwise_bottleneck_conv1d'] = nn.Conv1d(in_channels = out.shape[1], out_channels = self.k * out.shape[1], groups = out.shape[1], kernel_size = 3, stride = self.s, padding = 1)
        


        out = self.layer_dict['depthwise_bottleneck_conv1d'].forward(out)
        print("\n bottleneck depthwise 2nd layer ouput shape: ",out.shape)
        self.layer_dict['bn1'] = nn.BatchNorm1d(num_features= out.shape[1])

        out = self.layer_dict['bn1'].forward(out)
        out = F.relu(out)
        
        self.layer_dict['last_bottleneck_conv1d'] = nn.Conv1d(in_channels = out.shape[1], out_channels = self.c, kernel_size = 1, stride = self.s, padding = 0)
        self.layer_dict['bn2'] = nn.BatchNorm1d(num_features= self.c)
        
        out = self.layer_dict['last_bottleneck_conv1d'].forward(out)
        out = self.layer_dict['bn2'].forward(out)
        print("\n bottleneck 3rd layer ouput shape: ",out.shape)
        
        self.layer_dict['SE'] = SqueezeExcitation(self.c, 16)
        out = self.layer_dict['SE'](out)
        
        print("\n bottleneck ouput shape: ",out.shape)
        out = out + x

        return out
    
    def forward(self, x):
        out = x

        out = self.layer_dict['fist_bottleneck_conv1d'].forward(out)
        out = self.layer_dict['bn0'].forward(out)
        out = F.relu(out)

        out = self.layer_dict['depthwise_bottleneck_conv1d'].forward(out)
        out = self.layer_dict['bn1'].forward(out)
        out = F.relu(out)

        out = self.layer_dict['last_bottleneck_conv1d'].forward(out)
        out = self.layer_dict['bn2'].forward(out)

        out = self.layer_dict['SE'](out)
        
        out = out + x

        return out
    
    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

        self.logit_linear_layer.reset_parameters()


class Quite_A_Big_Model(nn.Module):
    def __init__(self, input_shape, hidden_units, output_classes):
        """
        Initializes a fully connected network for tabular data.
        :param input_features: Number of input features (columns in the dataset).
        :param hidden_units: Number of units in the hidden layer.
        :param output_classes: Number of output classes (e.g., N and A).
        """
        super(Quite_A_Big_Model, self).__init__()
        self.input_shape = input_shape
        self.fc1 = nn.Linear(input_shape[-1], hidden_units)  # First layer
        self.fc2 = nn.Linear(hidden_units, output_classes)  # Output layer
        self.relu = nn.ReLU()  # Activation function
        self.softmax = nn.Softmax(dim=1)  # For multi-class classification
        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['fist_conv1d'] = nn.Conv1d(in_channels = out.shape[1], out_channels = 1, kernel_size = 1, stride = 2, padding = 0)
        
        out = self.layer_dict['fist_conv1d'].forward(out)

        self.layer_dict['bn0_outside'] = nn.BatchNorm1d(num_features= out.shape[1])
        out = self.layer_dict['bn0_outside'].forward(out)  #ERRROOOORRRRR
        out = F.relu(out)

        bottleneckRepeats = [1, 2, 2, 2, 2, 1]
        cList = [8, 16, 32, 64, 128, 256]
        for i in range(0, 6):
            ctr = 2
            while bottleneckRepeats[i] > 0:
                print("Bottleneck ",i)
                self.layer_dict[f'Bottleneck_{i}_{bottleneckRepeats}'] = Bottleneck(input_shape=out.shape, k = 6, c = cList[i], n = bottleneckRepeats[i], s = ctr )
                ctr = ctr - 1 # creates the bottleneck
            
                #runs the values through the created bottleneck as it was created
                out = self.layer_dict[f'Bottleneck_{i}_{bottleneckRepeats}'].forward(out)

    def forward(self, x):
        """
        Forward pass through the network.
        :param x: Input tensor.
        :return: Predicted logits.
        """
        out = self.layer_dict['fist_conv1d'].forward(x)
        out = self.layer_dict['bn0_outside'].forward(out)
        out = F.relu(out)

        bottleneckRepeats = [1, 2, 2, 2, 2, 1]
        cList = [8, 16, 32, 64, 128, 256]

        for i in range(0, 6):
            while bottleneckRepeats[i] > 0:
                out = self.layer_dict[f'Bottleneck_{i}_{bottleneckRepeats}'].forward(out)



        x = self.relu(self.fc1(out))
        x = self.softmax(self.fc2(x))

        return 

    def reset_parameters(self):
        """
        Resets the network parameters for reinitialization.
        """
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


