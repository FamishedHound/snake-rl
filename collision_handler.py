class collision():
    def __init__(self, apple, snake):
        self.apple = apple
        self.snake = snake

    def check_collision_apple_snake(self):

        if self.apple.x == self.snake.snake_head[0] and self.apple.y == self.snake.snake_head[1]:
            self.apple.spawn_apple()
            self.snake.add_segment(self.apple.x,self.apple.y)



            return True


