from State import state


class Q_table():
    def __init__(self, apple_pos, starting_pos):
        self.apple_pos = apple_pos
        self.starting_pos = starting_pos

        self.utilities = [{0:0}, {1:0},{2:0}, {3:0}]

    def update_table(self, action, new_utility):
        self.utilities[action][action]=new_utility
        self.utilities = self.utilities
        #print(f"{self.utilities} after")

    def get_all_actions_utilities(self):
        #print()
        #rint(self.utilities)
        return [value for x in self.utilities for k,value in x.items()]

    def get_all_keys(self):
        return [k for x in self.utilities for k,value in x.items()]
    def __repr__(self):
        return str(self.utilities)

    def __iter__(self):
        ''' Returns the Iterator object '''
        return utilityIterator(self)
    def __delitem__(self, key):
        del self.utilities[key]
    def __copy__(self):
        cls = self.__class__
        newobject = cls.__new__(cls)
        newobject.__dict__.update(self.__dict__)
        return newobject

class utilityIterator():
    def __init__(self, Q_table):
        self.Q_table = Q_table
        self.index = 0

    def __next__(self):
        if self.index < len(self.Q_table.utilities):
            result = self.Q_table.utilities[self.index]
            self.index += 1


            return result
        # End of Iteration
        raise StopIteration

