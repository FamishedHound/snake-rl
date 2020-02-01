from current_run import single_game


class games_manager():
    def __init__(self, height, width, snake):

        self.width = width
        self.height = height
        self.Q_tables = []
        self.snake = snake
        self.past_games = []
        self.eplsion_greedy_param = 0.1

    def look_up_memory_of_games(self, apple_pos, starting_pos):
        for game in self.past_games:
            if game.apple_pos == apple_pos and game.starting_pos == starting_pos:
                return game

        game = single_game(apple_pos, starting_pos, self.width, self.height, self.snake)
        self.past_games.append(game)
        return game

    def start_new_game(self, apple_pos, starting_pos):
        current_game = self.look_up_memory_of_games(apple_pos, starting_pos)
        current_game.new_game()
        return current_game
