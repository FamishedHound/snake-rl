from copy import deepcopy


class state():

    def __init__(self, table_pos, apple_pos):
        self.table_pos = table_pos
        self.apple_pos = apple_pos

        self.memory = []
        from Q_table import Q_table
        self.Q_table = Q_table(self.apple_pos)

    def get_Q_table(self):

        return self.Q_table



