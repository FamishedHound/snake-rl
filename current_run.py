from Q_table import Q_tables


class single_game():
    def __init__(self, apple_pos, starting_pos, width, height, snake):
        self.height = height
        self.width = width
        self.apple_pos = apple_pos
        self.starting_pos = starting_pos
        self.states = []
        self.snake = snake
        self.eplsion_greedy_param = 0.1
        self.Q_table = Q_tables(self.apple_pos, self.starting_pos, self.height, self.width)

    def make_a_move(self):

