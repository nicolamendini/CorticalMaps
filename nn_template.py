import torch
import torch.nn as nn

## general network class, where the network is simply defined by a list of layers
class Network(nn.Module):
    
    def __init__(self, h, device, bias=False):
        
        super(Network, self).__init__()
        
        # initialising the layers dictionary and storing architecture
        self.layers = {}
        self.h = h
        
        # filling in the layers dictionary
        for i in range(len(h)):
            
            layer_type = h[i][0]
                
            if layer_type=='dense':
                self.layers[i] = nn.Linear(h[i][2], h[i][1], bias=bias, device=device)
                
            elif layer_type=='conv2d':
                self.layers[i] = nn.Conv2d(
                    h[i-1][1], 
                    h[i][1], 
                    h[i][2], 
                    padding=h[i][2]//2, 
                    bias=bias,
                    device=device
                )
                
            elif layer_type=='pool2d':
                self.layers[i] = nn.MaxPool2d(h[i][2])
                
            elif layer_type=='upsample':
                self.layers[i] = nn.Upsample(scale_factor=h[i][2])
                
            elif layer_type=='flatten':
                self.layers[i] = nn.Flatten()
                
            elif layer_type=='unflatten':
                self.layers[i]= nn.Unflatten(1, (h[i][1], h[i][2], h[i][2]))
                
        print(self.layers.keys())    
                        
    def forward(self, x_b, debug=False):

        act = self.layers[0](x_b)
        
        # applying layer after layer
        for i in range(1, len(self.h)):
            
            if debug:
                print(i, act.shape)
            
            act = torch.relu(act)

            act = self.layers[i](act)

        return act
    


    # function to create a simmetric decoder provided an encoder architecture
    # it simply inverts all the layers so that they do no have to be rewritten
    def reverse_network(hidden):

        hidden = list(reversed(hidden))

        for i in range(len(hidden)):

            layer_tuple = hidden[i]
            layer_type = layer_tuple[0]

            if layer_type=='unflatten':
                hidden[i] = ('flatten', *layer_tuple[1:])

            elif layer_type=='flatten':
                hidden[i] = ('unflatten', *layer_tuple[1:])

            elif layer_type=='dense':
                hidden[i] = ('dense', layer_tuple[2], layer_tuple[1])

            elif layer_type=='pool2d':
                hidden[i] = ('upsample', hidden[i-1][1], layer_tuple[2])

            elif layer_type=='upsample':
                hidden[i] = ('pool2d', hidden[i-1][1], layer_tuple[2])

        return hidden

