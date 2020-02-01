import random

from Q_table import Q_table
from State import state


class single_game():
    def __init__(self, apple_pos, starting_pos, width, height, snake):
        self.height = height
        self.width = width
        self.apple_pos = apple_pos
        self.starting_pos = starting_pos
        self.states_memory = []
        self.snake = snake
        self.path_in_current_game = []
        self.state_list_current = None
        self.alpha = 0.1
        self.discount_factor = 0.99

        self.cumulative_utility = 0

    def initate_states(self):

        for dict in self.states_memory:
            for key, value in dict.items():
                if key == (self.apple_pos, self.starting_pos):
                    return value

        states = [state((x, y), self.apple_pos, self.starting_pos) for x in range(self.height) for y in range(self.width)]
        self.states_memory.append({(self.apple_pos, self.starting_pos): states})
        return states

    def new_game(self):
        self.cumulative_utility=0
        self.path_in_current_game = []
        self.state_list_current = self.initate_states()

    def get_current_state(self):
        for state in self.state_list_current:
            if self.snake.snake_head_x == state.table_pos[0] and self.snake.snake_head_y == state.table_pos[1]:
                return state

    def decide_action(self, epsilon_greedy,reward):
        if reward != -1:
            current_state = self.get_current_state()
            Q_table_cur = self.get_current_state().Q_table

            utilities = Q_table_cur.get_all_actions_utilities()

            if self.checkEqual(utilities):
                action = random.randint(0,3)
            else:
                roulette = random.uniform(0, 1)
                if roulette > epsilon_greedy:
                    action = max(utilities)
                else:
                    utilities.remove(max(utilities))
                    action = random.choice(utilities)
            self.update_previous_state(action, reward)
            self.path_in_current_game.append(current_state)

            return action
        else:
            self.update_previous_state(0, reward)
            return 0

    def update_previous_state(self, action, reward):
        if len(self.path_in_current_game) > 0:

            previous_state = self.path_in_current_game[-1]
            print("{} {}".format(action,previous_state.Q_table.utilities))
            a = previous_state.Q_table.utilities[action][action]
            b = max(
                previous_state.Q_table.get_all_actions_utilities())
            self.cumulative_utility += self.alpha * a + (1 - self.alpha) * \
                                       reward + self.discount_factor * b
            previous_state.Q_table.utilities[action][action] = self.cumulative_utility



    def checkEqual(self, lst):
        return lst[1:] == lst[:-1]
