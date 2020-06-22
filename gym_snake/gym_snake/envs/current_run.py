import pickle
import random

from gym_snake.gym_snake.envs.State import state


class single_game():
    def __init__(self, apple_pos, starting_pos, width, height, snake,load_from_file):
        self.states_memory = []

        if load_from_file[0]:
            print("loaded Q tables from a file {}".format(load_from_file[1]))
            self.states_memory = pickle.load(open(f"saved_models/{load_from_file[1]}.p", "rb"))
        self.height = height
        self.width = width
        self.apple_pos = apple_pos
        self.starting_pos = starting_pos

        self.snake = snake

        self.path_in_current_game = []
        self.state_list_current = []

        self.alpha = 0.1
        self.discount_factor = 0.90
        self.states = None
        self.cumulative_utility = 0
        self.previous_action = []

    def initate_states(self, apple_pos):
        if len(self.states_memory )> 0:
            for settings in self.states_memory:
                for apple_positions,states in settings.items():
                    if apple_positions == apple_pos:

                        return states
        a = [ state((x, y),apple_pos)  for x in range(self.height) for y in
             range(self.width)]

        self.states_memory.append({apple_pos:a})

        return a

    def new_game(self, height, width, apple_pos, starting_pos, snakes_pos):

        self.cumulative_utility = 0
        self.snake_pos = snakes_pos
        self.path_in_current_game = []
        self.previous_action = []
        self.state_list_current = self.initate_states(apple_pos)

    def get_current_state(self, snake_pos):

        for state in self.state_list_current:
            if snake_pos[0] == state.table_pos[0] and snake_pos[1] == state.table_pos[1]:
                return state

    def decide_action(self, epsilon_greedy, reward, snake_pos):

        if reward != -1:
            current_state = self.get_current_state(snake_pos)
            Q_table_cur = self.get_current_state(snake_pos).get_Q_table()


            utilities = Q_table_cur.get_all_actions_utilities()

            if self.checkEqual(utilities):

                action = random.randint(0, 3)
            else:

                roulette = random.uniform(0, 1)
                if roulette > epsilon_greedy:
                    x = self.find_max_value_in_Q_table(Q_table_cur)

                    action = random.choice(list(x))
                else:

                    action = random.choice(self.find_exploration_indexes(Q_table_cur))
            if len(self.previous_action) > 0:
                self.update_previous_state(action, self.previous_action[-1], reward, current_state)
            self.path_in_current_game.append(current_state)
            self.previous_action.append(action)  # change us we are not fixed

            return action  # change us we are not fixed
        else:

            self.set_for_out_of_bound(self.previous_action[-1], reward)
            #print("lost")

    def update_previous_state(self, action, previous_action, reward, current_state):
        previous_state = self.path_in_current_game[-1]
        if len(self.path_in_current_game) > 0 and reward != 1:

            a = current_state.Q_table.utilities[action][action]
            b = previous_state.Q_table.utilities[previous_action][previous_action]
            c = max(current_state.Q_table.get_all_actions_utilities())

            self.cumulative_utility = self.alpha * b + (1 - self.alpha) * (reward + self.discount_factor * c)

            previous_state.Q_table.update_table(previous_action, self.cumulative_utility)
        else:

            previous_state.Q_table.update_table(previous_action, 1)

    def set_for_out_of_bound(self, previous_action, reward):
        previous_state = self.path_in_current_game[-1]
        if len(self.path_in_current_game) > 0:
            b = previous_state.Q_table.utilities[previous_action][previous_action]
            c = max(previous_state.Q_table.get_all_actions_utilities())
            #print(self.discount_factor * c + b)
            self.cumulative_utility = self.alpha * -1 + (1 - self.alpha) * (b + (self.discount_factor * c) + reward)
            previous_state.Q_table.update_table(previous_action, reward)

    def get_key_by_value(self, dic, value):
        for k, v in dic.items():
            if v == value:
                # print(k)
                pass

    def checkEqual(self, lst):
        return lst[1:] == lst[:-1]

    def find_max_value_in_Q_table(self, lst):

        dic_index = 0
        maxs = -20
        maxs_list = []
        # print(lst)
        # print()
        for index, dic in enumerate(lst):
            # print(dic)
            for key, value in dic.items():
                if value == maxs:
                    maxs_list.append(key)
                if value > maxs:
                    maxs_list = []
                    dic_index = key

                    maxs = value
                    maxs_list.append(dic_index)

        if len(maxs_list) == 0:
            return [dic_index]
        else:
            return maxs_list

    def find_exploration_indexes(self, lst):
        maxes = self.find_max_value_in_Q_table(lst)
        indices = []
        to_ignore = []
        # print()
        for index in sorted(maxes, reverse=True):
            to_ignore.append(index)
        for dic in lst:
            # print(to_ignore)
            for k, v in dic.items():
                if k not in to_ignore:
                    indices.append(k)
        return indices

    def save_model(self,filename):
        pickle.dump(self.states_memory, open(f"saved_models/{filename}.p", "wb"))
