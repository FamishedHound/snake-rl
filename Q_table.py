from State import state


class Q_table():
    def __init__(self, apple_pos, starting_pos):
        self.apple_pos = apple_pos
        self.starting_pos = starting_pos


        self.utilities = [{0: 0}, {1: 0}, {2: 0}, {3: 0}]
