import random
from collections import deque
import numpy as np

class replay_memory():
    def __init__(self, capacity):
        self.memory = deque()
        self.capacity = capacity

    def append(self, experience):
        if len(self.memory) > self.capacity:
            self.memory.popleft()

        self.memory.append(experience)
    def sample(self,batch_size):

        sampling = random.sample(self.memory,batch_size)

        return sampling

