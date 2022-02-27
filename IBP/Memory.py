import torch
import torch.nn as nn
import util

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, c_size=1, hidden_size=100, output_size=50):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        #self.linear = nn.Linear(hidden_size,output_size)
        self.hidden = (torch.zeros(1,1,self.hidden_size),
                       torch.zeros(1,1,self.hidden_size))
        self.cell_state = torch.zeros(1, c_size)

    def forward(self, route, actual_state, last_imagined_state, action, new_state, reward, j, k, prev_c):
        seq = []
        seq.append(route)
        seq.append(actual_state)
        seq.append(last_imagined_state)
        seq.append(action)
        seq.append(new_state)
        seq.append(reward)
        seq.append(j)
        seq.append(k)
        seq_tensor = util.tensor_from(seq)
        #seq, self.hidden = self.lstm(seq_tensor.view(len(seq_tensor),1,-1), self.hidden)
        #out = self.linear(c.view(len(seq),-1))
        c, self.cell_state = self.lstm_cell(seq_tensor.unsqueeze(0), 
                                            (prev_c.unsqueeze(0),
                                            self.cell_state))
        return c.squeeze() # Return whole sequence (c_i)
