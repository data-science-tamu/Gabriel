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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define elasticity network

class ElasticityNet(nn.Module):
    def __init__(self):
        super(ElasticityNet, self).__init__()
        self.fc1 = nn.Linear(2, num_neuron)
        self.fc2 = nn.Linear(num_neuron, num_neuron)
        self.fc3 = nn.Linear(num_neuron, num_neuron)
        self.fc4 = nn.Linear(num_neuron, num_neuron)
        self.fc5 = nn.Linear(num_neuron, num_neuron)
        self.fc6 = nn.Linear(num_neuron, num_neuron)
        self.fc7 = nn.Linear(num_neuron, num_neuron)
        self.fc8 = nn.Linear(num_neuron, num_neuron)
        self.fc9 = nn.Linear(num_neuron, num_neuron)
        self.fc10 = nn.Linear(num_neuron, num_neuron)
        self.fc11 = nn.Linear(num_neuron, num_neuron)
        self.fc12 = nn.Linear(num_neuron, num_neuron)
        self.fc13 = nn.Linear(num_neuron, num_neuron)
        self.fc14 = nn.Linear(num_neuron, num_neuron)
        self.fc15 = nn.Linear(num_neuron, num_neuron)
        self.fc16 = nn.Linear(num_neuron, num_neuron)
        self.fc17 = nn.Linear(num_neuron, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        x = torch.relu(self.fc10(x))
        x = torch.relu(self.fc11(x))
        x = torch.relu(self.fc12(x))
        x = torch.relu(self.fc13(x))
        x = torch.relu(self.fc14(x))
        x = torch.relu(self.fc15(x))
        x = torch.relu(self.fc16(x))
        return self.fc17(x)


# Define displacement network

class DisplacementNet(nn.Module):
    def __init__(self):
        super(DisplacementNet, self).__init__()
        self.fc1 = nn.Linear(2, num_neuron)
        self.fc2 = nn.Linear(num_neuron, num_neuron)
        self.fc3 = nn.Linear(num_neuron, num_neuron)
        self.fc4 = nn.Linear(num_neuron, num_neuron)
        self.fc5 = nn.Linear(num_neuron, num_neuron)
        self.fc6 = nn.Linear(num_neuron, num_neuron)
        self.fc7 = nn.Linear(num_neuron, num_neuron)
        self.fc8 = nn.Linear(num_neuron, num_neuron)
        self.fc9 = nn.Linear(num_neuron, num_neuron)
        self.fc10 = nn.Linear(num_neuron, num_neuron)
        self.fc11 = nn.Linear(num_neuron, num_neuron)
        self.fc12 = nn.Linear(num_neuron, num_neuron)
        self.fc13 = nn.Linear(num_neuron, num_neuron)
        self.fc14 = nn.Linear(num_neuron, num_neuron)
        self.fc15 = nn.Linear(num_neuron, num_neuron)
        self.fc16 = nn.Linear(num_neuron, num_neuron)
        self.fc17 = nn.Linear(num_neuron, 1)

    def forward(self, x):
        x = torch.nn.functional.silu(self.fc1(x))
        x = torch.nn.functional.silu(self.fc2(x))
        x = torch.nn.functional.silu(self.fc3(x))
        x = torch.nn.functional.silu(self.fc4(x))
        x = torch.nn.functional.silu(self.fc5(x))
        x = torch.nn.functional.silu(self.fc6(x))
        x = torch.nn.functional.silu(self.fc7(x))
        x = torch.nn.functional.silu(self.fc8(x))
        x = torch.nn.functional.silu(self.fc9(x))
        x = torch.nn.functional.silu(self.fc10(x))
        x = torch.nn.functional.silu(self.fc11(x))
        x = torch.nn.functional.silu(self.fc12(x))
        x = torch.nn.functional.silu(self.fc13(x))
        x = torch.nn.functional.silu(self.fc14(x))
        x = torch.nn.functional.silu(self.fc15(x))
        return self.fc17(x)


# Initialize models, optimizer, and loss function

elasticity_net = ElasticityNet().to(device)
displacement_net = DisplacementNet().to(device)
optimizer = optim.Adam(list(elasticity_net.parameters()) + list(displacement_net.parameters()), lr=learn_rate)
criterion = nn.L1Loss()

x_elas = torch.tensor(x_elas, dtype=torch.float32).to(device)
x_disp = torch.tensor(x_disp, dtype=torch.float32).to(device)
y_disp = torch.tensor(y_disp, dtype=torch.float32).to(device)
y_elas = torch.tensor(y_elas, dtype=torch.float32).to(device)

# Training process

start_time = time.time()

for i in range(200001):
    elasticity_net.train()
    displacement_net.train()

    optimizer.zero_grad()

    y_pred_m = elasticity_net(x_elas).squeeze()
    y_pred_v = displacement_net(x_disp).squeeze()

    # Define losses based on strain and displacement

    loss_m = torch.abs(y_pred_m.mean() - y_elas.mean())
    loss_v = torch.abs(y_pred_v.mean() - y_disp[:, 0].mean())
    loss = loss_m + loss_v

    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(f"Step {i}: Loss = {loss.item()}")

y_pred_m_value = y_pred_m.detach().cpu().numpy()
y_pred_v_value = y_pred_v.detach().cpu().numpy()

np.savetxt('y_pred_m_final', y_pred_m_value)
np.savetxt('y_pred_v_final', y_pred_v_value)

print("--- %s Elapsed time ---" % (time.time() - start_time))
