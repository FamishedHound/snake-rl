import math


class collision():
    def __init__(self, apple, snake):
        self.apple = apple
        self.snake = snake

    def return_reward(self,height,width):

        if self.apple.x == self.snake.snake_head_x and self.apple.y == self.snake.snake_head_y:

            self.snake.add_segment(self.apple.x,self.apple.y)


            return 10
        if self.snake.snake_head_x<0 or self.snake.snake_head_x >height-1:

            return -1
        if self.snake.snake_head_y < 0 or self.snake.snake_head_y > width-1:
            return -1
        for segment in self.snake.segments:
            if self.snake.snake_head_x==segment.x and self.snake.snake_head_y==segment.y:
                return -1

        # if self.snake.snake_head[0] != 1:
        #     return -1
        #print(f"{self.snake.snake_head_x} , {self.apple.x} , {self.snake.snake_head_y} ,{self.apple.y} ")
        # print(self.calculateDistance(self.snake.snake_head[0],self.snake.snake_head[1],self.apple.x,self.apple.y))
        #print(1/self.calculateDistance(self.snake.snake_head[0],self.snake.snake_head[1],self.apple.x,self.apple.y))
        return -0.1

    def calculateDistance(self,x1, y1, x2, y2):
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist


