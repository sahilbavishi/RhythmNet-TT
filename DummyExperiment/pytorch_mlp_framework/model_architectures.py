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
        print(out.shape)
        self.layer_dict['fist_bottleneck_conv1d'] = nn.Conv1d(in_channels = out.shape[1], out_channels = self.c, kernel_size = 1, stride = self.s, padding = 0)
        self.layer_dict['bn0'] = nn.BatchNorm1d(num_features= self.c)
        print(out.shape)
        out = self.layer_dict['fist_bottleneck_conv1d'].forward(out)
        out = self.layer_dict['bn0'].forward(out)
        out = F.relu(out)
        print(out.shape)

        self.layer_dict['depthwise_bottleneck_conv1d'] = nn.Conv1d(in_channels = out.shape[1], out_channels = self.k * out.shape[1], groups = out.shape[1], kernel_size = 3, stride = self.s, padding = 1)
        


        out = self.layer_dict['depthwise_bottleneck_conv1d'].forward(out)
        print(out.shape)
        self.layer_dict['bn1'] = nn.BatchNorm1d(num_features= out.shape[1])

        out = self.layer_dict['bn1'].forward(out)
        out = F.relu(out)
        print(out.shape)
        self.layer_dict['last_bottleneck_conv1d'] = nn.Conv1d(in_channels = out.shape[1], out_channels = self.c, kernel_size = 1, stride = self.s, padding = 0)
        self.layer_dict['bn2'] = nn.BatchNorm1d(num_features= self.c)
        print(out.shape)
        out = self.layer_dict['last_bottleneck_conv1d'].forward(out)
        out = self.layer_dict['bn2'].forward(out)

        self.layer_dict['SE'] = SqueezeExcitation(self.c, 16)
        out = self.layer_dict['SE'](out)
        
        print(out.shape)
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

class Quite_A_Big_Model(nn.Module):
    def __init__(self, input_features, hidden_units, output_classes, conv_block):
        """
        Initializes a fully connected network for tabular data.
        :param input_features: Number of input features (columns in the dataset).
        :param hidden_units: Number of units in the hidden layer.
        :param output_classes: Number of output classes (e.g., N and A).
        """
        super(Quite_A_Big_Model, self).__init__()
        self.fc1 = nn.Linear(input_features, hidden_units)  # First layer
        self.fc2 = nn.Linear(hidden_units, output_classes)  # Output layer
        self.relu = nn.ReLU()  # Activation function
        self.softmax = nn.Softmax(dim=1)  # For multi-class classification

        
        self.layer_dict = nn.ModuleDict()


    def forward(self, x):
        """
        Forward pass through the network.
        :param x: Input tensor.
        :return: Predicted logits.
        """


        out = x
        out = self.layer_dict['input_conv'].forward(out)


        # x = self.relu(self.fc1(x))
        # x = self.softmax(self.fc2(x))
        return x

    def reset_parameters(self):
        """
        Resets the network parameters for reinitialization.
        """
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


