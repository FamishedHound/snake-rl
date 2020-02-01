from current_run import single_game


class games_manager():
    def __init__(self,height,width,snake):

        self.width = width
        self.height = height
        self.Q_tables =[]
        self.snake = snake
        self.past_games =[]
        self.eplsion_greedy_param = 0.1




    def start_a_new_game(self,apple_pos,starting_pos):
        for dict in self.past_games:
            for key, value in dict.items():
                if key == (apple_pos, starting_pos):
                    return value

        games = [single_game((x, y), apple_pos, self.starting_pos) for x in self.height for y in self.width]
        self.states_memory.append({(self.apple_pos, self.starting_pos): states})
        return states

