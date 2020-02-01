class collision():
    def __init__(self, apple, snake):
        self.apple = apple
        self.snake = snake

    def return_reward(self,height,width):

        if self.apple.x == self.snake.snake_head[0] and self.apple.y == self.snake.snake_head[1]:
            self.apple.spawn_apple()
            self.snake.add_segment(self.apple.x,self.apple.y)



            return 1
        if self.snake.snake_head_x<0 or self.snake.snake_head_x >height-1:
            return -1
        if self.snake.snake_head_y < 0 or self.snake.snake_head_y > width-1:
            return -1
        return 0


