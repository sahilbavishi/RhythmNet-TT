import torch
import torch.nn as nn
import torch.nn.functional as F

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
