import torch.nn as nn
import torch.nn.functional as F

class ManagerModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=3):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        
    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        return X

print(sum([1,  # highest norm is always included
            1 if True else 0,
            1 if True else 0,
            100 if False else 0,
        ]))