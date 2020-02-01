from Q_table import Q_table


class state():

    def __init__(self, table_pos, apple_pos, starting_pos):
        self.table_pos = table_pos
        self.apple_pos = apple_pos
        self.starting_pos = starting_pos
        self.current_table = self.select_a_table()
        self.memory = []

    def select_a_table(self):
        for q_table in self.memory:

            if q_table.apple_pos == self.apple_pos and q_table.starting_pos == self.starting_pos:
                return q_table

        q_table = Q_table(self.apple_pos, self.starting_pos)
        self.memory.append(q_table)
        return q_table
