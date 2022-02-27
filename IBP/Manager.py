import torch.nn as nn
import torch.nn.functional as F

class ManagerModel(nn.Module):
    def __init__(self, context_size=1, hidden_size=(100,100), output_size=3):
        super().__init__()
        self.linear1 = nn.Linear(context_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], output_size)
        self.activation = nn.ReLU()
        
    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = F.relu(self.linear3(X))
        return X

