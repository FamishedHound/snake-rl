import random


class f_approximation():
    def __init__(self, epsilon_param):
        self.no_action = 4
        self.no_weights = 5
        self.weights = self.create_actions_and_weights(self.no_weights, self.no_action)
        self.epsilon_para = epsilon_param

        self.discount_factor = 0.95
        self.learning_rate = 0.01
        self.alpha = 0.9

        self.previous_decision = None
        self.previous_utility = None

    def create_actions_and_weights(self, how_many_weights, no_action):
        list_of_actions = []
        for _ in range(no_action):
            action_0 = [random.uniform(0, 1) for _ in range(how_many_weights)]
            list_of_actions.append(action_0)

        return list_of_actions

    def make_utilities_out_of_state_vector(self, apple_pos, snake_pos, no_actions):
        utilities = []
        feature_vec = [apple_pos[0], apple_pos[1], snake_pos[0], snake_pos[1]]
        for index, weight in enumerate(self.weights):
            utility = 0
            for counter in range(no_actions):
                utility = feature_vec[counter] + weight[counter]
            utility = utility + weight[-1]
            utilities.append(utility)

        return utilities

    def make_decision(self, apple_pos, snake_pos,reward):

        utilities = self.make_utilities_out_of_state_vector(apple_pos, snake_pos, self.no_action)

        if self.epsilon_greedy(self.epsilon_para):
            argmax = max(utilities)
            self.previous_utility = argmax
            decision = self.find_index_in_the_list(argmax, utilities)

        else:

            decision = random.choice(range(4))
            self.previous_utility = utilities[decision]

        self.previous_decision = decision
        return decision

    def epsilon_greedy(self, epsilon_param):
        eps = random.uniform(0, 1)
        return True if eps > epsilon_param else False

    def update_weights(self,utilities,reward,current_argmax,decision):
        if self.previous_decision !=None:
            td_part = reward+self.discount_factor*current_argmax - self.previous_utility

            weights_to_update = self.weights[decision]







    def find_index_in_the_list(self, value_to_find, utilities):
        for x in enumerate(utilities):
            if x == value_to_find:
                return x
