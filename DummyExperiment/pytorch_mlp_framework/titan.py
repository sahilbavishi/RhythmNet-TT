import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NeuralMemory(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(NeuralMemory,self).__init__()
        self.input_dim=input_dim
        self.hidden_units=hidden_units
        self.BuildModule()

    def BuildModule(self):
        x = torch.zeros(self.input_shape)
        print("Building module with input shape ", x.shape)
        self.layer_dict=nn.ModuleDict()
        self.layer_dict["Input_Layer"] = nn.Linear(x.shape[-1], self.hidden_units)
        x = self.layer_dict["Input_Layer"].forward(x)
        print("shape after 1st layer: ", x.shape)
        self.layer_dict["bn0"] = nn.BatchNorm1d(x.shape[1])
        x = self.layer_dict["bn0"].forward(x)
        self.layer_dict["Output_Layer"] = nn.Linear(x.shape[-1], self.input_shape[-1]) # this might give an error because of how python works, any change in x might change input because of python memory storage
        x = self.layer_dict["Output_Layer"].forward(x)
        print("shape after final layer: ", x.shape)
        print("final x shape: ", x.shape)
        return x
    
    def forward(self, input):
        '''
        Used for concatenation purposes and at the time of predicting post attention
        '''
        x=input
        out = self.layer_dict["Input_Layer"].forward(x)
        out = F.relu(self.layer_dict["bn0"].forward(out))
        out = self.layer_dict["Output_Layer"].forward(out)
        return out
    
    def forward_inferenece(self, input):
        '''
        Used to train the model with post attention sequence I assume we need to write derivatives here
        '''
        x=input
        out = self.layer_dict["Input_Layer"].forward(x)
        out = F.relu(self.layer_dict["bn0"].forward(out))
        out = self.layer_dict["Output_Layer"].forward(out)
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

class Attention(nn.Module):
    def __init__(self, input):
        '''
        Concatenate the Original Sequence with the output of the neural memory and then pass it here.
        Returns self attention on the Concatenated Sequence
        '''


class Titan(nn.Module):
    def __init__(self, input):
        '''
        x = input sequence [post backbone]
        layer_dict['Neural_Memory'] = call Neural Memory
        out = layer_dict['Neural_Memory'].forward(x)

        x = out + x

        layer_dict['Attention'] = call Attention
        x = layer_dict['Attention'].forward(x)

        temp_out = layer_dict['Neural_Memory'].forward_inference(x)  # I guess it learns here
        
        out = layer_dict['Neural_Memory'].forward(x)  #after learning new context

        output = x dot out

        return output [ this then goes into the ffn ]

        '''