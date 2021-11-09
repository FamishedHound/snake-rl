import torch.nn
import torch

from Manager import ManagerModel
from Controller import ControllerModel
from Memory import LSTMModel
from DQN.DQN_agent import DQN_agent

proj_path = "C:\\Users\\killi\\Documents\\Repositories\\snake-rl\\"
# torch.save(model.state_dict(), proj_path + f"new_models\\GAN13_{generator_amplifier}_{discriminator_deamplifier}_new.pt")
class IBP(object):
    def __init__(self, env, img_layers):
        self.manager = ManagerModel()
        self.controller = DQN_agent(action_number=4, frames=1, learning_rate=0.0001, discount_factor=0.99, batch_size=8,
                                   epsilon=1, save_model=False, load_model=True,
                                   path="C:\\Users\\killi\\Documents\\Repositories\\snake-rl\\DQN_trained_model\\10x10_model_with_tail.pt",
                                   epsilon_speed=1e-4)
        self.GAN = torch.load(proj_path + f"new_models\\GAN13_3_15_new.pt")
        # self.reward_predictor = ### Our best rew.pred. 
        self.memory = LSTMModel()
    
    def select_route(self):
        pass

    def select_action(self):
        pass

    def run(self):

        pass
