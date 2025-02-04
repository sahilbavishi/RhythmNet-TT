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

class FeedForwardModule(nn.Module):
    def __init__(self, input_shape,hidden_units,num_classes,finallayer="softmax"):
        """Type of feedforward network that has 2 layers, and you can define using softmax or just a regression model"""
        super(FeedForwardModule,self).__init__()
        self.input_shape=input_shape
        self.hidden_units=hidden_units
        self.num_classes=num_classes
        self.finallayer=finallayer
        self.BuildModule()
    def BuildModule(self):
        x=torch.zeros(self.input_shape)
        print("Building module with input shape ", x.shape)
        self.layer_dict=nn.ModuleDict()
        self.layer_dict["FF0"]=nn.Linear(x.shape[-1],self.hidden_units)
        x=self.layer_dict["FF0"].forward(x)
        print("shape after 1st layer: ",x.shape)
        self.layer_dict["bn0"]=nn.BatchNorm1d(x.shape[1])
        x=self.layer_dict["bn0"].forward(x)
        print("shape after batch normalisation layer: ", x.shape)
        output_num=1
        if self.finallayer=="softmax":
            #Change final layer size to number of classes, if this is a classification task
            output_num=self.num_classes
        self.layer_dict["FF1"]=nn.Linear(x.shape[-1],output_num)
        x=self.layer_dict["FF1"].forward(x)
        print("shape after final layer: ",x.shape)
        print("final x shape: ",x.shape)
        return x
    def forward(self,input):
        x=input
        x=self.layer_dict["FF0"].forward(x)
        x=F.relu(self.layer_dict["bn0"].forward(x))
        x=self.layer_dict["FF1"].forward(x)

        if self.finallayer=="softmax":
            x=F.softmax(x,dim=2)
        return x

class MultipleOutputFFN(nn.Module):
    def __init__(self, input_dim,output_dim):
        """Final layer of model - takes input and produces classes of output_dim along with a last input of position"""
        super(MultipleOutputFFN,self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.BuildModule()
    def BuildModule(self):
        x=torch.zeros(self.input_dim)
        self.layer_dict=nn.ModuleDict()
        self.layer_dict["Locations"]=FeedForwardModule(self.input_dim,12,self.output_dim,finallayer="not")
        out1=self.layer_dict["Locations"].forward(x)
        print("output1 shape: ",out1.shape)
        self.layer_dict["Classification"]=FeedForwardModule(self.input_dim,12,self.output_dim)
        out2=self.layer_dict["Classification"].forward(x)
        print("output2 shape: ",out2.shape)
        outfinal=torch.cat([out1,out2],dim=2)
        print("output final shape: ",outfinal.shape)
        return outfinal
    def forward(self,input):
        x=input
        out1=self.layer_dict["Locations"].forward(x)
        out2=self.layer_dict["Classification"].forward(x)
        outfinal=torch.cat([out1,out2],dim=2)
        return outfinal

class SqueezeExcite(nn.Module):
    def __init__(self, input_shape,reduction=16):
        """Squeeze and Excitement module for 1 dimensional data"""
        super(SqueezeExcite,self).__init__()
        self.input_shape=input_shape
        self.red=reduction
        self.BuildModule()
    def BuildModule(self):
        out=torch.zeros(self.input_shape)
        self.layer_dict=nn.ModuleDict()
        bottleneck=max(out.shape[1]//self.red,1)
        self.layer_dict["Avg_Pooling"]=nn.AvgPool1d(1)
        self.layer_dict["Conv1D"]=nn.Conv1d(out.shape[1],bottleneck,kernel_size=1,padding=0)
        self.layer_dict["Relu"]=nn.ReLU(True)
        self.layer_dict["Conv1d2"]=nn.Conv1d(bottleneck,out.shape[1],kernel_size=1,padding=0)
        self.layer_dict["Sigmoid"]=nn.Sigmoid()
    def forward(self,input):
        out=input
        out=self.layer_dict["Avg_Pooling"].forward(out)
        print(out.shape)
        out=self.layer_dict["Conv1D"].forward(out)
        out=self.layer_dict["Relu"].forward(out)
        out=self.layer_dict["Conv1d2"].forward(out)
        out=self.layer_dict["Sigmoid"].forward(out)
        return out*input




class SEModule(nn.Module):
    def __init__(self, channels, divide=4):
        super(SEModule, self).__init__()
        bottleneck = channels //  divide
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottleneck(nn.Module):
    def __init__(self, input_shape, k, c, n, s):
        super(Bottleneck,self).__init__()
        self.input_shape = input_shape
        self.k = k
        self.c = c
        self.n = n
        self.s = s
        self.build_module()

    def build_module(self):

        print("------------------------------------------")

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

        self.layer_dict['depthwise_bottleneck_conv1d'] = nn.Conv1d(in_channels = out.shape[1], out_channels = self.k * out.shape[1], groups = out.shape[1], kernel_size = 3, stride = 1, padding = 0)
        


        out = self.layer_dict['depthwise_bottleneck_conv1d'].forward(out)
        print("\n bottleneck depthwise 2nd layer ouput shape: ",out.shape)
        self.layer_dict['bn1'] = nn.BatchNorm1d(num_features= out.shape[1])

        out = self.layer_dict['bn1'].forward(out)
        out = F.relu(out)
        
        self.layer_dict['last_bottleneck_conv1d'] = nn.Conv1d(in_channels = out.shape[1], out_channels = self.c, kernel_size = 1, stride = 1, padding = 1)
        self.layer_dict['bn2'] = nn.BatchNorm1d(num_features= self.c)
        
        out = self.layer_dict['last_bottleneck_conv1d'].forward(out)
        out = self.layer_dict['bn2'].forward(out)


        print("Output before SE",out.shape)

        out = out.unsqueeze(-1)

        print("Output before SE after change",out.shape)
        self.layer_dict['SE'] = SqueezeExcitation(input_channels = self.c, squeeze_channels = 4)
        out = self.layer_dict['SE'](out)

        out = out.squeeze(-1)
        
        print("\n bottleneck ouput shape: ",out.shape)

        if(out.shape == x.shape):
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
        
        out = out.unsqueeze(-1)
        out = self.layer_dict['SE'](out)
        out = out.squeeze(-1)

        if(out.shape == x.shape):
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





class cnnBackbone(nn.Module):
    def __init__(self, input_shape, hidden_units, output_classes, d_model):
        """
        Initializes a fully connected network for tabular data.
        :param input_features: Number of input features (columns in the dataset).
        :param hidden_units: Number of units in the hidden layer.
        :param output_classes: Number of output classes (e.g., N and A).
        """
        super(cnnBackbone, self).__init__()
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.output_classes = output_classes
        self.d_model = d_model
        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)
        out = x

        # first conv
        self.layer_dict['fist_conv1d'] = nn.Conv1d(in_channels = out.shape[1], out_channels = 1, kernel_size = 1, stride = 2, padding = 0)
        
        out = self.layer_dict['fist_conv1d'].forward(out)

        self.layer_dict['bn0_outside'] = nn.BatchNorm1d(num_features= out.shape[1])
        out = self.layer_dict['bn0_outside'].forward(out)  #ERRROOOORRRRR
        out = F.relu(out)

        # bottlenecks
        bottleneckRepeats = [1, 2, 2, 2, 2, 1]
        strideList=[2, 2, 2, 2, 2, 1]
        cList = [8, 16, 32, 64, 128, 256]
        for i in range(0, 6):
            ctr = strideList[i]
            while bottleneckRepeats[i] > 0:
                print(f'Bottleneck_{i}_{bottleneckRepeats[i]}')
                self.layer_dict[f'Bottleneck_{i}_{bottleneckRepeats[i]}'] = Bottleneck(input_shape=out.shape, k = 6, c = cList[i], n = bottleneckRepeats[i], s = ctr )
                ctr = ctr - 1 # creates the bottleneck
                #runs the values through the created bottleneck as it was created
                out = self.layer_dict[f'Bottleneck_{i}_{bottleneckRepeats[i]}'].forward(out)
                bottleneckRepeats[i]-=1

        # last conv
        self.layer_dict['last_conv1d'] = nn.Conv1d(in_channels = out.shape[1], out_channels = self.d_model, kernel_size = 1, stride = 1, padding = 0)
        
        out = self.layer_dict['last_conv1d'].forward(out)

        self.layer_dict['bn1_outside'] = nn.BatchNorm1d(num_features= out.shape[1])
        out = self.layer_dict['bn1_outside'].forward(out)
        out = F.relu(out)

        print("Output shape: ",out.shape)

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
        # cList = [8, 16, 32, 64, 128, 256]

        for i in range(0, 6):
            while bottleneckRepeats[i] > 0:
                out = self.layer_dict[f'Bottleneck_{i}_{bottleneckRepeats[i]}'].forward(out)
                bottleneckRepeats[i]-=1


        # last conv
        out = self.layer_dict['last_conv1d'].forward(out)
        out = self.layer_dict['bn1_outside'].forward(out)
        out = F.relu(out)

        return 

    def reset_parameters(self):
        """
        Resets the network parameters for reinitialization.
        """
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


