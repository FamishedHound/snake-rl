import random
from copy import deepcopy

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
        self.state_list_current = self.initate_states()
        self.alpha = 0.1
        self.discount_factor = 0.90
        self.states = None
        self.cumulative_utility = 0
        self.previous_action = []

    def initate_states(self):

        a = [state((x, y), self.apple_pos, self.starting_pos) for x in range(self.height) for y in
             range(self.width)]
        return a

    def new_game(self, height, width, apple_pos, starting_pos, snakes_pos):

        self.cumulative_utility = 0
        self.snake_pos = snakes_pos
        self.path_in_current_game = []
        self.previous_action=[]
    def get_current_state(self,snake_pos):
        # print("{} inside".format(self.snake_pos))
        for state in self.state_list_current:
            if snake_pos[0] == state.table_pos[0] and snake_pos[1] == state.table_pos[1]:
                return state

    def decide_action(self, epsilon_greedy, reward,snake_pos):
        print()
        if reward!=-1 :
            current_state = self.get_current_state(snake_pos)
            Q_table_cur = self.get_current_state(snake_pos).get_Q_table()
            print("QTable for state {} with pos {} ".format(Q_table_cur,snake_pos))


            utilities = Q_table_cur.get_all_actions_utilities()

            if self.checkEqual(utilities):

                action = random.randint(0, 3)
            else:

                roulette = random.uniform(0, 1)
                if roulette > epsilon_greedy:
                    x = self.find_max_value_in_Q_table(Q_table_cur)
                    #print(" x {}".format(x))
                    action = random.choice(list(x))
                else:

                    action = random.choice(self.find_exploration_indexes(Q_table_cur))
            if len(self.previous_action) > 0:
                self.update_previous_state(action,self.previous_action[-1], reward,current_state)
            self.path_in_current_game.append(current_state)
            self.previous_action.append(action) #change us we are not fixed
            return action #change us we are not fixed
        else:

            self.set_for_out_of_bound(self.previous_action[-1], reward)
            print("lossing condition {}".format(snake_pos))




    def update_previous_state(self, action,previous_action, reward,current_state):
        previous_state = self.path_in_current_game[-1]
        if len(self.path_in_current_game) > 0 and reward!=1:

            #print(action)
            #print(previous_state.Q_table.utilities)
            a = current_state.Q_table.utilities[action][action]
            b = previous_state.Q_table.utilities[previous_action][previous_action]
            c = max(current_state.Q_table.get_all_actions_utilities())

            self.cumulative_utility =  self.alpha*b + (1-self.alpha)*(reward + self.discount_factor*c)


            previous_state.Q_table.update_table(previous_action, self.cumulative_utility)
        else:

            previous_state.Q_table.update_table(previous_action, 1)
    def set_for_out_of_bound(self,previous_action,reward):
        previous_state = self.path_in_current_game[-1]
        if len(self.path_in_current_game) > 0:


            b = previous_state.Q_table.utilities[previous_action][previous_action]
            c = max(previous_state.Q_table.get_all_actions_utilities())
            print(self.discount_factor*c+b)
            self.cumulative_utility = self.alpha * -1 + (1 - self.alpha) *( b + (self.discount_factor * c) + reward)
            previous_state.Q_table.update_table(previous_action, reward)



    def get_key_by_value(self, dic, value):
        for k, v in dic.items():
            if v == value:
                #print(k)
                pass

    def checkEqual(self, lst):
        return lst[1:] == lst[:-1]

    def find_max_value_in_Q_table(self, lst):

        dic_index = 0
        maxs = -20
        maxs_list = []
        #print(lst)
        #print()
        for index, dic in enumerate(lst):
            #print(dic)
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
        #print()
        for index in sorted(maxes, reverse=True):
            to_ignore.append(index)
        for dic in lst:
            #print(to_ignore)
            for k, v in dic.items():
                if k not in to_ignore:
                    indices.append(k)
        return indices




