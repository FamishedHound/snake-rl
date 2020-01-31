from State import state


class Q_tables():
    def __init__(self,apple_pos,starting_pos,height,width):
        self.apple_pos = apple_pos
        self.starting_pos = starting_pos
        self.height = height
        self.width = width


        self.states =[]

    def initate_states(self):
        self.states = [state((x,y)) for x in self.height for y in self.width]
