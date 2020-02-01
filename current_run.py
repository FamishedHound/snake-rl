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



    def initate_states(self):

        for dict in self.states_memory:
            for key, value in dict.items():
                if key == (self.apple_pos, self.starting_pos):
                    return value

        states = [state((x, y), self.apple_pos, self.starting_pos) for x in self.height for y in self.width]
        self.states_memory.append({(self.apple_pos, self.starting_pos): states})
        return states
    def new_game(self):

        self.current_state = self.initate_states()