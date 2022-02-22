import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=100, output_size=50):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size,output_size)
        self.hidden = (torch.zeros(1,1,self.hidden_size),
                       torch.zeros(1,1,self.hidden_size))

    def forward(self, route, actual_state, last_imagined_state, action, new_state, reward, i_action, i_imagination):
        c, self.hidden = self.lstm(seq.view(len(seq),1,-1), self.hidden)
        pred = self.linear(c.view(len(seq),-1))
        return pred[-1]
