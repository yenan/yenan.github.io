import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def gen_sin_waves(period, seq_len, num_seq):
    # sample random starting positions between -2*period and 2*period
    start = np.random.randint(-2*period, 2*period, (num_seq, 1))  
    # for each starting position, create a sequence of consecutive time steps
    x = start + np.arange(seq_len)
    # compute the value sequences
    data = np.sin(x / period).astype('float64')
    return data

class SinLSTM(nn.Module):
    def __init__(self):
        super(SinLSTM, self).__init__()
        # LSTM cell with 1D input and 50D hidden state
        self.lstm = nn.LSTMCell(1, 50) 
        # output layer takes in the 50D hidden state and output a 1D output 
        self.linear = nn.Linear(50, 1)

    def forward(self, x, future=0): 
        outputs = []
        # initial hidden state and cell state set to 0
        h_t = torch.zeros(x.size(0), 50, dtype=torch.double).to(x.device)
        c_t = torch.zeros(x.size(0), 50, dtype=torch.double).to(x.device)

        # predict outputs for inputs
        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]

        # predict future
        for i in range(future):
            h_t, c_t = self.lstm(output, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

if __name__ == '__main__':
    # set random seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    # generate value sequences
    period = 10
    seq_len = 200
    num_seq = 500
    data = torch.from_numpy(gen_sin_waves(period, seq_len, num_seq)).double().to(device)

    # sequences 4 to 500 are used as the training sequences
    data_tr = data[3:,:]
    x_train = data_tr[:, :-1]
    y_train = data_tr[:, 1:]

    # sequences 1 to 3 are used as the test sequences
    data_ts = data[:3,:]
    x_test = data_ts[:, :150]
    y_test = data_ts[:, 150:]  

    # build the model
    seq = SinLSTM().double().to(device)
    criterion = nn.MSELoss()

    # initialize optimizer 
    optimizer = optim.SGD(seq.parameters(), lr=.1, momentum=0.9)

    # train
    for i in range(10):
        optimizer.zero_grad()
        out = seq(x_train)
        loss = criterion(out, y_train)
        print('[%5d: %5.5f]' % (i,loss.item()))
        loss.backward()
        optimizer.step()
