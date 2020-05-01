from kaggle_environments import evaluate, make
import gym
from IPython.core.display import display, HTML
class ConnectX:
    def __init__(self):
        env = make("connectx", debug=True)

        print(env.reset())
        env.reset()
        # Play as the first agent against "negamax" agent.
        env.run(['random', 'random'])
        #env.run([my_agent, "negamax"])
        out = env.render(mode="ansi", width=500, height=450)
        print(out)

if __name__ == '__main__':
    connect = ConnectX()