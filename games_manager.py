from current_run import single_game


class games_manager():
    def __init__(self,height,width,snake):

        self.width = width
        self.height = height
        self.Q_tables =[]
        self.snake = snake




    def start_a_game(self,apple_pos,starting_pos):
        for game in self.Q_tables:

            if game.apple_pos == apple_pos and starting_pos == starting_pos:
                return game

        game = single_game(apple_pos, starting_pos, self.height, self.width,self.snake)
        self.Q_tables.append(game)
        return game

