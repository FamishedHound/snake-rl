class state():

    def __init__(self, table_pos):
        self.table_pos = table_pos
        self.utility = 0

    def update_utility(self, new_util):
        self.utility = new_util
        