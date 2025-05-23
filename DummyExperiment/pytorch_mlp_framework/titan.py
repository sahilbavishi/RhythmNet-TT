import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NeuralMemory(nn.Module):
    def __init__(self, input_shape, hidden_units, alpha=0.1, nu = 0.9, theta = 0.3):
        super(NeuralMemory,self).__init__()
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.accumulated_surprise = None
        self.alpha = alpha # past memory to forget
        self.nu = nu # surprise decay (how quickly past “surprise” fades)
        self.theta = theta # momentary surprise scaling
        self.BuildModule()

    def BuildModule(self):
        # If input_shape is an int, assume it represents the feature dimension and add a batch dimension of 1.
        if isinstance(self.input_shape, int):
            x = torch.zeros(1, self.input_shape)
        else:
            x = torch.zeros(self.input_shape)
        print("Building module with input shape ", x.shape)

        self.layer_dict=nn.ModuleDict()
        self.layer_dict["Input_Layer"] = nn.Linear(x.shape[-1], self.hidden_units)
        x = self.layer_dict["Input_Layer"].forward(x)
        print("shape after 1st layer: ", x.shape)
        self.layer_dict["bn0"] = nn.BatchNorm1d(x.shape[1])
        x = self.layer_dict["bn0"].forward(x)

        out_features = self.input_shape if isinstance(self.input_shape, int) else self.input_shape[-1]
        self.layer_dict["Output_Layer"] = nn.Linear(x.shape[-1], out_features)
        x = self.layer_dict["Output_Layer"](x)
        print("shape after final layer: ", x.shape)
        return x
    
    def forward(self, x):
        '''
        Used for concatenation purposes and at the time of predicting post attention
        '''
        out = self.layer_dict["Input_Layer"].forward(x)
        out = F.relu(self.layer_dict["bn0"].forward(out))
        out = self.layer_dict["Output_Layer"].forward(out)
        return out
    
    def get_params(self):
        return [p for p in self.parameters() if p.requires_grad]
    
    def surprise_score(self, k, v):
        prediction = self.forward(k)
        loss = F.mse_loss(prediction, v)

        params = self.get_params()
        surprise_s = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        return surprise_s

    def update_memory(self, surprise):
        """
        Update memory parameters: w -> (1-alpha)w + alpha*surprise
        This is a differentiable version that allows gradients to flow through
        """
        updated_params = {}
        for (name, param), grad in zip(self.named_parameters(), surprise):
            updated_params[name] = (1 - self.alpha) * param + self.alpha * grad
        return updated_params

    def forward_inference(self, k, q, v):
        '''
        Used to train the model with post attention sequence
        k: keys (input)
        q: query
        v: values (target)
        '''
        momentary_surprise = self.surprise_score(k, v)

        # Initialize accumulated_surprise if it's the first call
        if self.accumulated_surprise is None:
            self.accumulated_surprise = [torch.zeros_like(grad) for grad in momentary_surprise]
        
        # Update accumulated_surprise element-wise
        for i, (acc_surp, mom_surp) in enumerate(zip(self.accumulated_surprise, momentary_surprise)):
            self.accumulated_surprise[i] = (self.nu * acc_surp) - (self.theta * mom_surp)
        
        # Update accumulated surprise element-wise (differentiable)
        new_accumulated = []
        for acc, mom in zip(self.accumulated_surprise, momentary_surprise):
            new_accumulated.append(self.nu * acc - self.theta * mom)
        self.accumulated_surprise = new_accumulated

        updated_params = self.update_memory(self.accumulated_surprise)

        # compute updated parameters as a function of the current parameters and the computed gradients
        output = torch.func.functional_call(self, updated_params, q) 
        return output

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

class TryBackprop(nn.Module):
    def __init__(self, input_shape, hidden_units):
        super(TryBackprop,self).__init__()
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.BuildModule()
    
    def BuildModule(self):
        # two linear layers with NeuralMemory in the middle, quries, keys and values
        x = torch.zeros(self.input_shape)
        self.layer_dict=nn.ModuleDict()
        self.layer_dict["Queries"] = nn.Linear(x.shape[-1], self.hidden_units)
        q = self.layer_dict["Queries"].forward(x)
        self.layer_dict["Keys"] = nn.Linear(x.shape[-1], self.hidden_units)
        k = self.layer_dict["Keys"].forward(x)
        self.layer_dict["Values"] = nn.Linear(x.shape[-1], self.hidden_units)
        v = self.layer_dict["Values"].forward(x)

        self.layer_dict["Neural_Memory"] = NeuralMemory(input_shape=q.shape, hidden_units=self.hidden_units)
        out = self.layer_dict["Neural_Memory"].forward_inference(k, q, v)

        self.layer_dict["Output_Layer"] = nn.Linear(out.shape[-1], self.input_shape[-1])
        out = self.layer_dict["Output_Layer"].forward(out)
        return out
    
    def forward(self, x):
        q = self.layer_dict["Queries"].forward(x)
        k = self.layer_dict["Keys"].forward(x)
        v = self.layer_dict["Values"].forward(x)
        out = self.layer_dict["Neural_Memory"].forward_inference(k, q, v)
        out = self.layer_dict["Output_Layer"].forward(out)
        return out


'''
BELOW IS THE ATTENTION MODEL, I AM USING TO THE SAME AS THE PREVIOUS TRANSFORMER.
'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Initialize the PositionalEncoding module.

        Args:
            d_model: The number of expected features in the input.
            max_len: The maximum length of the input sequence.
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # Create a matrix to hold the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension for easier integration
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to the input.
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model].
        Returns:
            Tensor with positional encodings added to the input.
        """
        x = x + self.pe[:x.size(0), :]
        return x

    def reset_parameters(self):
        """
        Resets the parameters (if any learnable parameters are added in future versions).
        """
        self.pe.zero_()


# Transformer adapted for our task
class ECGTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_queries, num_encoder_layers, num_decoder_layers, max_seq_len=5000):
        super(ECGTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.max_seq_len = max_seq_len
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_seq_len)

        # Transformer components
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        
        # Learnable object queries
        self.object_queries = nn.Parameter(torch.rand(num_queries, d_model))
        
    def forward(self, features):
        """
        :param features: Extracted features from ECG signals (batch_size, seq_len, d_model)
        :return: Predicted classes and positions of heartbeats
        """
        # Add positional encoding
        features = self.positional_encoding(features)  # Shape: (batch_size, seq_len, d_model)

        # Encoder
        encoded_features = self.encoder(features)  # Shape: (batch_size, seq_len, d_model)
        
        # Prepare object queries for each sample in the batch
        batch_size = features.size(0)
        queries = self.object_queries.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (batch_size, num_queries, d_model)
        
        # Decoder
        decoded_features = self.decoder(queries, encoded_features)  # Cross-attention with encoder output
        
        return decoded_features

class Attention(nn.Module):
    def __init__(self, input_shape ,d_model,transformer_heads,N_q=6):
        '''
        Concatenate the Original Sequence with the output of the neural memory and then pass it here.
        Returns self attention on the Concatenated Sequence
        '''
        super(Attention,self).__init__()
        self.input_shape=input_shape
        self.d_model=d_model
        if d_model/transformer_heads!=d_model//transformer_heads:
            #If the d_model is not a multiple of transformer heads, return an error and exit
            raise ValueError("Error: the d_model is not a multiple of the number of transformer heads")
        self.num_heads=transformer_heads
        self.N_q=N_q
        self.build_module()

    def build_module(self):
        print(f"Input Shape to attention block: {self.input_shape}")
        out=torch.zeros(self.input_shape)
        self.layer_dict=nn.ModuleDict()
        self.layer_dict["Transformer"]=ECGTransformer(d_model=out.shape[-1],num_heads=self.num_heads,num_queries=out.shape[1],num_encoder_layers=4,num_decoder_layers=4)
        out=self.layer_dict["Transformer"].forward(out)
        print(f"Output Shape after attention block: {out.shape}")
    def forward(self, input):
        # input = input.unsqueeze(1)
        out=self.layer_dict["Transformer"].forward(input)
        return out


class Titan(nn.Module):
    def __init__(self, input_shape , d_model,transformer_heads, hidden_units, num_classes, N_q=6):
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

        super(Titan, self).__init__()
        self.input_shape=input_shape
        self.d_model=d_model
        if d_model/transformer_heads!=d_model//transformer_heads:
            #If the d_model is not a multiple of transformer heads, return an error and exit
            raise ValueError("Error: the d_model is not a multiple of the number of transformer heads")
        self.transformer_heads=transformer_heads
        self.hidden_units=hidden_units
        self.num_classes = num_classes
        self.N_q=N_q
        self.build_module()

    def build_module(self):
        out = torch.zeros(self.input_shape)
        self.layer_dict=nn.ModuleDict()
        print(f"Input shape before neural memory: {out.shape}")
        self.layer_dict['Neural_Memory'] = NeuralMemory(input_shape = self.input_shape, hidden_units = self.hidden_units)
        out1 = self.layer_dict['Neural_Memory'].forward(out)
        out = torch.cat((out1, out), dim=1)
        print(f"Output shape after neural memory: {out.shape}")
        self.layer_dict['Attention'] = Attention(input_shape = out.shape, d_model = self.d_model, transformer_heads = self.transformer_heads, N_q = self.N_q)
        out = self.layer_dict['Attention'].forward(out)
        print(f"Output shape after Attention: {out.shape}")
        return out

    def forward(self, input):
        out1 = self.layer_dict['Neural_Memory'].forward(input)
        out = torch.cat((out1, input), 1)
        out = self.layer_dict['Attention'].forward(out)
        '''
        Gotta add the inference part here and then do the dot product. I do not know how to do this.
        '''
        return out