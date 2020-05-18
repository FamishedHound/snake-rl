from DQN.DQN_agent import DQN_agent

from kaggle_environments import evaluate, make, utils
import matplotlib.pyplot as plt
from gym_snake.gym_snake.envs import Board

dqn_agent = DQN_agent(action_number=4, frames=2, learning_rate=0.001, discount_factor=0.99, batch_size=32,
                      epsilon=1, save_model=False, load_model=True,
                      path="C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\DQN_trained_model\\10x10_model_with_tail_new.pt",
                      epsilon_speed=1e-4)
flag = True
env = Board(4, 4,control="dqn")
env.render()
observation, reward = env.reset()

while True:



    action = dqn_agent.make_action(observation, reward)
    # plt.imshow(observation)
    # plt.show()
    observation, reward, done, _ = env.step(action)
    # plt.imshow(observation)
    # plt.show()
    # print()
    if done:
        observation,reward = env.reset()
        # plt.imshow(observation)
        # plt.show()
        # print()