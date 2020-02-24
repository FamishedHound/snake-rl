import random


class f_approximation():
    def __init__(self, epsilon_param):
        self.no_action = 4
        self.no_weights = 5
        self.weights = self.create_actions_and_weights(self.no_weights, self.no_action)
        self.epsilon_para = epsilon_param

        self.discount_factor = 0.95
        self.learning_rate = 0.001
        self.alpha = 0.9

        self.previous_decision = None
        self.previous_utility = 0
        self.previous_feature_vec = None

    def create_actions_and_weights(self, how_many_weights, no_action):
        list_of_actions = []
        for _ in range(no_action):
            action_0 = [random.uniform(0, 1) for _ in range(how_many_weights)]
            list_of_actions.append(action_0)

        return list_of_actions

    def make_utilities_out_of_state_vector(self, apple_pos, snake_pos,feature_vec):
        utilities = []

        for weight in self.weights:
            utility = 0
            for counter in range(len(feature_vec)):
                utility += feature_vec[counter] * weight[counter]
            utility = utility + weight[-1]
            utilities.append(utility)

        return utilities

    def make_decision(self, apple_pos, snake_pos,reward):
        feature_vec = [apple_pos[0], apple_pos[1], snake_pos[0], snake_pos[1]]
        utilities = self.make_utilities_out_of_state_vector(apple_pos, snake_pos,feature_vec)
        current_max = max(utilities)

        if self.epsilon_greedy(self.epsilon_para):

            temp = current_max
            decision = self.find_index_in_the_list(current_max, utilities)

        else:

            decision = random.choice(range(3))
            temp = utilities[decision]

        self.update_weights(reward, current_max, decision, feature_vec)
        self.previous_utility = temp
        self.previous_decision = decision
        self.previous_feature_vec = feature_vec
        return decision

    def epsilon_greedy(self, epsilon_param):
        eps = random.uniform(0, 1)
        return True if eps > epsilon_param else False

    def update_weights(self, reward, current_max, decision, feature_vec):
        if self.previous_decision !=None:
            td_part = reward+self.discount_factor*current_max - self.previous_utility

            weights_to_update = self.weights[decision]
            new_weights =[]
            for index,f in enumerate(feature_vec):
                new_weights.append(weights_to_update[index] + self.learning_rate * td_part * f)

            new_weights.append(weights_to_update[-1] + self.learning_rate * td_part)
            self.weights[decision] = new_weights
        print(self.weights)


    def restart(self):
        self.previous_decision = None
        self.previous_utility = 0
        self.previous_feature_vec = None


    def find_index_in_the_list(self, value_to_find, utilities):
        for index,x in enumerate(utilities):
            if x == value_to_find:
                return index
