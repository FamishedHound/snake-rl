from copy import deepcopy

from current_run import single_game


class games_manager():
    def __init__(self,apple_pos,starting_pos, height, width, snake):

        self.width = width
        self.height = height
        self.Q_tables = []
        self.snake = snake
        self.past_games = []
        self.eplsion_greedy_param = 0.1



        self.game = single_game(apple_pos, starting_pos, self.width, self.height, self.snake)



    def start_new_game(self, apple_pos, starting_pos,snake_pos):
        current_game = self.game
        current_game.new_game(self.height,self.width,apple_pos,starting_pos,snake_pos)
        return current_game
