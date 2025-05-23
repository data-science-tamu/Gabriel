import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, layers, activation, in_tf=None, out_tf=None):
        super().__init__()
        self.activation = activation
        self.linears = nn.ModuleList()
        self.in_tf = in_tf
        self.out_tf = out_tf
        
        # Define Cp and k as trainable parameters
        self.Cp = nn.Parameter(torch.tensor(0.75, dtype=torch.float32, requires_grad=True))  # Initial guess
        self.k = nn.Parameter(torch.tensor(8.5, dtype=torch.float32, requires_grad=True))  # Initial guess

        # Weight initialization
        for i in range(1, len(layers)):
            self.linears.append(nn.Linear(layers[i-1], layers[i]))
            nn.init.xavier_uniform_(self.linears[-1].weight)
            nn.init.zeros_(self.linears[-1].bias)
                      
    def forward(self, inputs):
        X = inputs
        # Input transformation
        if self.in_tf:
            X = self.in_tf(X)
        # Linear layers    
        for linear in self.linears[:-1]:
            X = self.activation(linear(X))
        # Last layer, no activation
        X = self.linears[-1](X)
        # Output transformation
        if self.out_tf:
            X = self.out_tf(X)
        return X
