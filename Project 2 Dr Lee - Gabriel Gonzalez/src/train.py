import torch
import torch.nn as nn
import time
import os
import matplotlib.pyplot as plt

def loss(f,target=None):
    if target == None:
        return torch.sum(torch.square(f))/f.shape[0]
    if isinstance(target,float):
        return torch.sum(torch.square(f-target))/f.shape[0]
    else:
        return nn.MSELoss()(f,target)

import os

    
def train(net, PDE, BC, point_sets, flags, iterations=50000, lr=5e-4, info_num=100, 
          test_in=None, test_out=None, w=[1., 1., 1., 1.], inv_params=[], x_min=[0, 0, 0], x_max=[40, 10, 6]):
    """
    Main training function with added visualization.
    """
    if inv_params == []:
        params = net.parameters()
    else:
        params = list(net.parameters()) + inv_params
    optimizer = torch.optim.Adam(params, lr=lr)

    n_bc, n_ic, n_pde, n_data = 0, 0, 0, 0
    for points, flag in zip(point_sets, flags):
        if flag[0] == 'BC':
            n_bc += points.shape[0]
        if flag[0] == 'IC':
            n_ic += points.shape[0]
        if flag[0] == 'domain':
            n_pde += points.shape[0]
        if flag[0] == 'data':
            n_data += points.shape[0]

    start_time = time.time()
    l_history = []
    err_history = [] if test_in is not None else None

    for epoch in range(iterations):
        optimizer.zero_grad()
        l_bc, l_ic, l_pde, l_data = 0, 0, 0, 0

        for points, flag in zip(point_sets, flags):
            if flag[0] == 'BC':
                f = BC(points[:, 0:1], points[:, 1:2], points[:, 2:3], points[:, 3:4], net, flag[1])
                l_bc += loss(f) * points.shape[0] / n_bc
            if flag[0] == 'IC':
                pred = net(points)
                l_ic += loss(pred, flag[1]) * points.shape[0] / n_ic
            if flag[0] == 'domain':
                f = PDE(points[:, 0:1], points[:, 1:2], points[:, 2:3], points[:, 3:4], net)
                l_pde += loss(f) * points.shape[0] / n_pde

        cost = (w[0] * l_bc + w[1] * l_ic + w[2] * l_pde) / 3
        l_history.append([cost.item(), l_bc.item(), l_ic.item(), l_pde.item()])

        if epoch % info_num == 0:
            elapsed = time.time() - start_time
            test_err = None
            if test_in is not None:
                with torch.no_grad():
                    T_pred = net(test_in)
                    test_err = loss(T_pred, test_out)
                    err_history.append(test_err.item())

            print(f"It: {epoch}, Loss: {cost:.3e}, BC: {l_bc:.3e}, IC: {l_ic:.3e}, PDE: {l_pde:.3e}, Time: {elapsed:.2f}")

            if test_in is not None:
                print(f", Test: {test_err:.3e}")



        cost.backward()
        optimizer.step()

    return l_history, err_history


def train2D(net,PDE,BC,point_sets,flags,iterations=50000,lr=5e-4,info_num=100,
         test_in = None, test_out=None,w=[1.,1.,1.,1.],inv_params=None):
    
    if inv_params == None:
        params = net.parameters()
    else:
        params = (list(net.parameters())+inv_params)
    optimizer = torch.optim.Adam(params,lr=lr)
    
    n_bc = 0
    n_ic = 0
    n_PDE = 0
    n_data =0
    for points,flag in zip(point_sets,flags):
        if flag[0] == 'BC':
            n_bc += points.shape[0]
        if flag[0] == 'IC':
            n_ic += points.shape[0]
        if flag[0] == 'domain':
            n_PDE += points.shape[0]
        if flag[0] == 'data':
            n_data += points.shape[0]
            
    start_time = time.time()
    
    l_history = []
    if test_in != None:
        err_history = []
    for epoch in range(iterations):
        optimizer.zero_grad()
        l_BC = 0
        l_IC = 0
        l_PDE = 0
        l_data = 0
    
        for points,flag in zip(point_sets,flags):
            if flag[0] == 'BC':
                f = BC(points[:,0:1],points[:,1:2],points[:,2:3],net,flag[1])
                l_BC += loss(f)*points.shape[0]/n_bc
            if flag[0] == 'IC':
                pred = net(points)
                l_IC += loss(pred,flag[1])*points.shape[0]/n_ic
            if flag[0] == 'data':
                pred = net(points)
                l_data += loss(pred,flag[1])*points.shape[0]/n_data
            if flag[0] == 'domain':
                f = PDE(points[:,0:1],points[:,1:2],points[:,2:3],net)
                l_PDE += loss(f)*points.shape[0]/n_PDE
            
        
        if n_data == 0:
            cost = (w[0]*l_BC+w[1]*l_IC+w[2]*l_PDE)/3 #weighted
            l_history.append([cost.item(),
                      l_BC.item(),
                      l_IC.item(),
                      l_PDE.item()])
            
            if epoch%info_num == 0:
                if test_in != None:
                    T_pred = net(test_in)
                    Test_err = loss(T_pred,test_out)
                    err_history.append(Test_err.item())
                    elapsed = time.time() - start_time
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Test: %.3e, Time: %.2f' 
                          % (epoch, cost, l_BC, l_IC, l_PDE, Test_err, elapsed))
                    start_time = time.time()
                else:
                    elapsed = time.time() - start_time
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Time: %.2f' 
                          % (epoch, cost, l_BC, l_IC, l_PDE,elapsed))
                    start_time = time.time()
                                        
        else:
            cost = (w[0]*l_BC+w[1]*l_IC+w[2]*l_PDE+w[3]*l_data)/4 #weighted
            l_history.append([cost.item(),
                              l_BC.item(),
                              l_IC.item(),
                              l_PDE.item(),
                              l_data.item()])
            
            if epoch%info_num == 0:
                if test_in != None:
                    T_pred = net(test_in)
                    Test_err = loss(T_pred,test_out)
                    err_history.append(Test_err.item())
                    elapsed = time.time() - start_time
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Data: %.3e, Test: %.3e, Time: %.2f' 
                          % (epoch, cost, l_BC, l_IC, l_PDE, l_data,Test_err, elapsed))
                    start_time = time.time()
                else:
                    elapsed = time.time() - start_time
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Data: %.3e, Time: %.2f' 
                          % (epoch, cost, l_BC, l_IC, l_PDE, l_data, elapsed))
                    start_time = time.time()
                
                if inv_params!=[]:
                    for value in inv_params:
                        print(value.item())
                    
            
        cost.backward() 
        optimizer.step()
        

    return l_history,err_history



def train2DnoBC(net,PDE,BC,point_sets,flags,iterations=50000,lr=5e-4,info_num=100,
         test_in = None, test_out=None,w=[1.,1.,1.,1.],inv_params=None):
    
    if inv_params == None:
        params = net.parameters()
    else:
        params = (list(net.parameters())+inv_params)
    optimizer = torch.optim.Adam(params,lr=lr)
    
    n_ic = 0
    n_PDE = 0
    n_data =0
    for points,flag in zip(point_sets,flags):
        if flag[0] == 'BC':
            n_bc += points.shape[0]
        if flag[0] == 'IC':
            n_ic += points.shape[0]
        if flag[0] == 'domain':
            n_PDE += points.shape[0]
        if flag[0] == 'data':
            n_data += points.shape[0]
            
    start_time = time.time()
    
    l_history = []
    if test_in != None:
        err_history = []
    for epoch in range(iterations):
        optimizer.zero_grad()
        l_BC = 0
        l_IC = 0
        l_PDE = 0
        l_data = 0
    
        for points,flag in zip(point_sets,flags):
            if flag[0] == 'BC':
                f = BC(points[:,0:1],points[:,1:2],points[:,2:3],net,flag[1])
                l_BC += loss(f)*points.shape[0]/n_bc
            if flag[0] == 'IC':
                pred = net(points)
                l_IC += loss(pred,flag[1])*points.shape[0]/n_ic
            if flag[0] == 'data':
                pred = net(points)
                l_data += loss(pred,flag[1])*points.shape[0]/n_data
            if flag[0] == 'domain':
                f = PDE(points[:,0:1],points[:,1:2],points[:,2:3],net)
                l_PDE += loss(f)*points.shape[0]/n_PDE
            
        
        if n_data == 0:
            cost = (w[0]*l_BC+w[1]*l_IC+w[2]*l_PDE)/3 #weighted
            l_history.append([cost.item(),
                      l_BC.item(),
                      l_IC.item(),
                      l_PDE.item()])
            
            if epoch%info_num == 0:
                if test_in != None:
                    T_pred = net(test_in)
                    Test_err = loss(T_pred,test_out)
                    err_history.append(Test_err.item())
                    elapsed = time.time() - start_time
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Test: %.3e, Time: %.2f' 
                          % (epoch, cost, l_BC, l_IC, l_PDE, Test_err, elapsed))
                    start_time = time.time()
                else:
                    elapsed = time.time() - start_time
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Time: %.2f' 
                          % (epoch, cost, l_BC, l_IC, l_PDE,elapsed))
                    start_time = time.time()
                                        
        else:
            cost = (w[0]*l_BC+w[1]*l_IC+w[2]*l_PDE+w[3]*l_data)/4 #weighted
            l_history.append([cost.item(),
                              l_BC.item(),
                              l_IC.item(),
                              l_PDE.item(),
                              l_data.item()])
            
            if epoch%info_num == 0:
                if test_in != None:
                    T_pred = net(test_in)
                    Test_err = loss(T_pred,test_out)
                    err_history.append(Test_err.item())
                    elapsed = time.time() - start_time
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Data: %.3e, Test: %.3e, Time: %.2f' 
                          % (epoch, cost, l_BC, l_IC, l_PDE, l_data,Test_err, elapsed))
                    start_time = time.time()
                else:
                    elapsed = time.time() - start_time
                    print('It: %d, Loss: %.3e, BC: %.3e, IC: %.3e, PDE: %.3e, Data: %.3e, Time: %.2f' 
                          % (epoch, cost, l_BC, l_IC, l_PDE, l_data, elapsed))
                    start_time = time.time()
                
                if inv_params!=[]:
                    for value in inv_params:
                        print(value.item())
                    
            
        cost.backward() 
        optimizer.step()
        

    return l_history,err_history