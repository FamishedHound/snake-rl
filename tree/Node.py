class Master():
    def __init__(self, img,past_action):
        self.parent = None
        self.img = img
        self.flag = True
        self.action = past_action


class Node():
    def __init__(self, parent, action, img):
        self.parent = parent
        self.action = action
        self.img = img

