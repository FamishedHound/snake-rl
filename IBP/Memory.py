import torch
import torch.nn as nn
import util

class LSTMModel(nn.Module):
    def __init__(self, input_size=21173, c_size=100, hidden_size=100, 
                       output_size=50, cuda_flag=True):
        super().__init__()
        self.cuda_flag = cuda_flag
        self.hidden_size = hidden_size
        if self.cuda_flag:
            self.lstm_cell = nn.LSTMCell(input_size, hidden_size).cuda()
            self.hidden = (torch.zeros(1,1,self.hidden_size).cuda(),
                       torch.zeros(1,1,self.hidden_size).cuda())
            self.cell_state = torch.zeros(1, c_size).cuda()
        else:
            self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
            self.hidden = (torch.zeros(1,1,self.hidden_size),
                       torch.zeros(1,1,self.hidden_size))
            self.cell_state = torch.zeros(1, c_size)

    def forward(self, route, actual_state, last_imagined_state, action, 
                      new_state, reward, j, k, prev_c):
        seq = []
        seq.append(route)
        seq.append(torch.flatten(torch.from_numpy(actual_state)))
        seq.append(torch.flatten(torch.from_numpy(last_imagined_state)))
        seq.append(action)
        seq.append(torch.flatten(torch.from_numpy(new_state)))
        seq.append(reward)
        seq.append(j)
        seq.append(k)
        seq_tensor = util.tensor_from(seq).float()

        if self.cuda_flag:
            seq_tensor = seq_tensor.cuda()
            prev_c = prev_c.cuda()

        c, self.cell_state = self.lstm_cell(seq_tensor.unsqueeze(0), 
                                            (prev_c.unsqueeze(0),
                                            self.cell_state))
        return c.squeeze() # Return whole sequence (c_i)
