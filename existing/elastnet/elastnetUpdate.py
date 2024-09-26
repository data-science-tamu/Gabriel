import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn import preprocessing

# Setup
num_neuron = 128
learn_rate = 0.001

x_disp = np.loadtxt('data_incompressible/m_rose_nu_05/disp_coord')
y_disp = np.loadtxt('data_incompressible/m_rose_nu_05/disp_data')
x_elas = np.loadtxt('data_incompressible/m_rose_nu_05/strain_coord')
y_elas = np.loadtxt('data_incompressible/m_rose_nu_05/m_data')

ss_x = preprocessing.StandardScaler()
x_disp = ss_x.fit_transform(x_disp.reshape(-1, 2))
x_elas = ss_x.fit_transform(x_elas.reshape(-1, 2))

# Define neural network architectures
class ElasticityNetwork(nn.Module):
    def __init__(self):
        super(ElasticityNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, num_neuron), nn.ReLU(),
            *[nn.Sequential(nn.Linear(num_neuron, num_neuron), nn.ReLU()) for _ in range(11)],
            nn.Linear(num_neuron, 1)
        )

    def forward(self, x):
        return self.layers(x)

class DisplacementNetwork(nn.Module):
    def __init__(self):
        super(DisplacementNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, num_neuron), nn.SiLU(),
            *[nn.Sequential(nn.Linear(num_neuron, num_neuron), nn.SiLU()) for _ in range(11)],
            nn.Linear(num_neuron, 1)
        )

    def forward(self, x):
        return self.layers(x)

elasticity_net = ElasticityNetwork()
disp_net = DisplacementNetwork()

# Define optimizer
optimizer = optim.Adam(list(elasticity_net.parameters()) + list(disp_net.parameters()), lr=learn_rate)

# Read displacements
y_u = torch.tensor(y_disp[:, 0], dtype=torch.float32)
y_disp_tensor = torch.tensor(y_disp, dtype=torch.float32)

# Calculate strains (define conv2d function)
def conv2d(x, weight):
    return nn.functional.conv2d(x, weight, stride=1, padding=0)

# Initialize conv weights (you will need to adapt this part according to your original logic)
conv_x = torch.tensor([[[-0.5]], [[-0.5]]], dtype=torch.float32)
conv_y = torch.tensor([[[0.5]], [[-0.5]]], dtype=torch.float32)

# Training process
start_time = time.time()
for i in range(200001):
    optimizer.zero_grad()
    
    # Forward pass
    x_elas_tensor = torch.tensor(x_elas, dtype=torch.float32)
    x_disp_tensor = torch.tensor(x_disp, dtype=torch.float32)
    
    y_pred_m = elasticity_net(x_elas_tensor)
    y_pred_v = disp_net(x_disp_tensor)

    # Calculate strains here (you'll need to implement these calculations)
    # For example, calculating fx_conv_sum_norm and fy_conv_sum_norm:
    # Placeholder calculations; replace these with your actual logic
    fx_conv_sum_norm = torch.randn_like(y_pred_m)  # Example tensor, replace with real calculation
    fy_conv_sum_norm = torch.randn_like(y_pred_v)  # Example tensor, replace with real calculation

    # Calculate loss
    mean_modu = torch.mean(torch.tensor(y_elas, dtype=torch.float32))
    loss_x = torch.mean(torch.abs(fx_conv_sum_norm))
    loss_y = torch.mean(torch.abs(fy_conv_sum_norm))
    loss_m = torch.abs(torch.mean(y_pred_m) - mean_modu)
    loss_v = torch.abs(torch.mean(y_pred_v))
    loss = loss_x + loss_y + loss_m / 100 + loss_v / 100
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    print(i, loss.item())

# Save predictions
np.savetxt('y_pred_m_final', y_pred_m.detach().numpy())
np.savetxt('y_pred_v_final', y_pred_v.detach().numpy())
print("--- %s Elapsed time ---" % (time.time() - start_time))
