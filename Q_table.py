from State import state


class Q_table():
    def __init__(self, apple_pos, starting_pos):
        self.apple_pos = apple_pos
        self.starting_pos = starting_pos

        self.utilities = [{0: 0}, {1: 0}, {2: 0}, {3: 0}]

    def update_table(self, action, new_utility):
        self.utilities[action][action]=new_utility

    def get_all_actions_utilities(self):

        return [value for x in self.utilities for k,value in x.items()]
