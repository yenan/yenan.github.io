import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0) 
    
    # load training and test data
    x, y = load_diabetes(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    x_train, x_test, y_train, y_test = [torch.from_numpy(t).float() 
            for t in [x_train, x_test, y_train, y_test]]
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # model architecture
    net = nn.Sequential(nn.Linear(x_train.shape[1], 100),
            nn.Sigmoid(),
            nn.Linear(100, 1),
            nn.Sigmoid(),
            nn.Linear(1, 100),
            nn.Sigmoid(),
            nn.Linear(100, 1),
            nn.ReLU())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    net = net.to(device)
    criterion = nn.MSELoss()
   
    # initialize optimizer 
    optimizer = optim.SGD(net.parameters(), lr=2, momentum=2)

    # initialize network parameters
    for param in net.parameters():
        nn.init.zeros_(param)

    # train
    batch_size = 2
    for epoch in range(10):
        perm = np.random.permutation(len(x_train))
        for b in range(int(len(x_train)/batch_size)):
            optimizer.zero_grad()
            indices = perm[b*batch_size:(b+1)*batch_size]
            x_batch = x_train[indices].to(device)
            y_batch = y_train[indices].to(device)
            out = net(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
        print('[%5d: %5.5f]' % (epoch, loss.item()))

    # test
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    out = net(x_test)
    loss = criterion(out, y_test)
    print('Test MSE: ', loss.item())
