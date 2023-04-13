import random
import time
import numpy as np
import math
from tkinter import Tk, Canvas, Frame, BOTH,Label
import threading
import torch
import torch.nn as nn


class BrainBase:
    """
    left=-1
    forward=0
    right=1
    """

    def __init__(self):
        self.output = 0
        self.input = [
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0
        ]
        self.midle=[]

    def predict(self, input_data):
        #print(input_data)
        if self.input[3] == 1:
            return 1
        return 0

    def pre_draw(self):
        return [15, 3]


class SimpleBrain(torch.nn.Module,BrainBase):
    def __init__(self):
        super(SimpleBrain,self).__init__()
        #BrainBase.__init__(self)
        self.linear1=nn.Linear(15,8)
        self.linear2=nn.Linear(8,3)
        self.midle=[[]]

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        self.midle[0]=x.tolist()
        x = torch.sigmoid(self.linear2(x))
        return x

    def pre_draw(self):
        return [15,8,3]

    def predict(self, input_data):

        output=self.forward(torch.FloatTensor(input_data))
        if output[0]>output[1]:
            if output[0]>output[2]:
                return -1
            else:
                return 1
        else:
            if output[1]>output[2]:
                return 0
            else:
                return 1

class N2Brain(torch.nn.Module,BrainBase):
    def __init__(self):
        super(SimpleBrain,self).__init__()
        #BrainBase.__init__(self)
        self.linear1=nn.Linear(15,8)
        self.linear2=nn.Linear(8,5)
        self.linear3=nn.Linear(5,3)
        self.midle=[[],[]]

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        self.midle[0]=x.tolist()
        x = torch.sigmoid(self.linear2(x))
        self.midle[1]=x.tolist()
        x = self.linear3(x)
        return x

    def pre_draw(self):
        return [15,8,5,3]

    def predict(self, input_data):

        output=self.forward(torch.FloatTensor(input_data))
        if output[0]>output[1]:
            if output[0]>output[2]:
                return -1
            else:
                return 1
        else:
            if output[1]>output[2]:
                return 0
            else:
                return 1



class Snake:
    """
    Decide:
    -1:sol
    0:ileri
    1:sağ

    Direction:
    0=Up
    1=Right
    2=Down
    3=Left
    """
    left = -1
    forward = 0
    right = 1

    def __init__(self, game, pos_x=0, pos_y=0, brain:BrainBase=None,cooldown=0.025):
        self.map = game.map
        self.game = game
        self.cooldown=cooldown
        self.pos_x = pos_x
        self.pos_y = pos_y
        # self.map[pos_y][pos_x]=3
        self.direction = 2
        self.gameover = False
        self.tail = []
        self.tail_size = 2
        self.set_head()
        if brain == None:
            self.brain = BrainBase()
        else:
            self.brain = brain
        #print(self.direction_to_point()) #TODO

    # TURN
    def single_turn(self):
        time.sleep(self.cooldown)
        self.brain.input = self.sens_all()
        decision = self.brain.predict(self.brain.input)
        self.calculate_direction(decision*2)
        self.brain.output = decision
        return self.forward()

    # TAIL
    def set_head(self):
        self.map[self.pos_y][self.pos_x] = Game.map_head
        self.game.update_ui(self.pos_x,self.pos_y)

    def add_tail(self):
        self.tail.append((self.pos_x, self.pos_y))
        self.map[self.pos_y][self.pos_x] = Game.map_tail
        self.game.update_ui(self.pos_x,self.pos_y)

    def remove_tail(self):
        self.map[self.tail[0][1]][self.tail[0][0]] = Game.map_empty
        self.game.update_ui(self.tail[0][0],self.tail[0][1])
        self.tail.pop(0)

    # MOVEMENTS
    def move_forward(self):
        return self.forward()

    def move_left(self):
        self.calculate_direction(Snake.left)
        return self.forward()

    def move_right(self):
        self.calculate_direction(Snake.right)
        return self.forward()

    def calculate_direction(self, direct):
        self.direction = (self.direction + direct) % 8

    def forward(self):
        direct = self.direction_to_point()
        new_position = (self.pos_x + direct[0], self.pos_y + direct[1])
        if self.game.make_move((self.pos_x, self.pos_y), new_position):
            self.add_tail()
            # self.tail.append((self.pos_x,self.pos_y))
            self.pos_x = new_position[0]
            self.pos_y = new_position[1]
            self.set_head()
            if len(self.tail) >= self.tail_size:
                self.remove_tail()
                # self.tail.pop(0)
        # print('poss:',(self.pos_x,self.pos_y),new_position)

    def direction_to_point(self):
        return (1 if self.direction == 2 else (-1 if self.direction == 6 else 0),
                -1 if self.direction == 0 else (1 if self.direction == 4 else 0))
        # SENSING

    def food_eaten(self):
        self.tail_size += 1
        return

    def find_direction(self, way):
        return (self.direction + way + 8) % 8

    def sens_all(self):
        data = [
            self.sens_obstacle(self.find_direction(-3)),
            self.sens_obstacle(self.find_direction(-2)),
            self.sens_obstacle(self.find_direction(-1)),
            self.sens_obstacle(self.find_direction(0)),
            self.sens_obstacle(self.find_direction(1)),
            self.sens_obstacle(self.find_direction(2)),
            self.sens_obstacle(self.find_direction(3)),

            0,
            0,
            0,
            0,
            0,
            0,
            0,
            self.tail_size
        ]
        food_data = self.sens_food()
        #print((self.pos_x,self.pos_y),(self.game.food_x,self.game.food_y))
        if food_data[0]!=-1:
            #print(((3+food_data[0]-self.direction)%7),food_data[1])
            data[7+((3+food_data[0]-self.direction)%7)]=10-food_data[1]

        return data

    # def sens_all(self):
    #    return [self.sens_obstacle(0),self.sens_obstacle(0.5),self.sens_obstacle(1),self.sens_obstacle(1.5),self.sens_obstacle(2),self.sens_obstacle(2.5),self.sens_obstacle(3),self.sens_obstacle(3.5)]
    def sens_food(self):
        if self.game.food_y == self.pos_y:
            if self.game.food_x >= self.pos_x:
                return (2, self.game.food_x - self.pos_x)
            else:
                return (6, self.pos_x - self.game.food_x)
        elif self.game.food_x == self.pos_x:
            if self.game.food_y >= self.pos_y:
                return (4, self.game.food_y - self.pos_y)
            else:
                return (0, self.pos_y - self.game.food_y)
        elif abs(self.game.food_x - self.pos_x) == abs(self.game.food_y - self.pos_y):
            if (self.game.food_x - self.pos_x) > 0 and (self.game.food_y - self.pos_y) < 0:
                return (1, (self.game.food_x - self.pos_x))
            elif (self.game.food_x - self.pos_x) > 0 and (self.game.food_y - self.pos_y) > 0:
                return (3, (self.game.food_x - self.pos_x))
            elif (self.game.food_x - self.pos_x) < 0 and (self.game.food_y - self.pos_y) > 0:
                return (5, (self.game.food_y - self.pos_y))
            elif (self.game.food_x - self.pos_x) < 0 and (self.game.food_y - self.pos_y) < 0:
                return (7, -(self.game.food_y - self.pos_y))
        else:
            return (-1, -1)

    def sens_obstacle(self, direction):
        raycast = 0
        if direction == 0:  # UP
            y = self.pos_y - 1
            while y >= 0:
                if (self.map[y][self.pos_x] >= 0):
                    raycast += 1
                else:
                    return raycast
                y -= 1
            return raycast
        elif direction == 1:  # UP-RIGHT
            y = self.pos_y - 1
            x = self.pos_x + 1
            while y >= 0 and x >= 0 and y < len(self.map) and x < len(self.map[self.pos_y]):
                if (self.map[y][x] >= 0):
                    raycast += 1
                else:
                    return raycast
                y -= 1
                x += 1
            return raycast
        elif direction == 2:  # RIGHT
            x = self.pos_x + 1
            while x < len(self.map[self.pos_y]):
                if (self.map[self.pos_y][x] >= 0):
                    raycast += 1
                else:
                    return raycast
                x += 1
            return raycast
        elif direction == 3:  # DOWN-RIGHT
            y = self.pos_y + 1
            x = self.pos_x + 1
            while y >= 0 and x >= 0 and y < len(self.map) and x < len(self.map[self.pos_y]):
                if (self.map[y][x] >= 0):
                    raycast += 1
                else:
                    return raycast
                y += 1
                x += 1
            return raycast
        if direction == 4:  # DOWN
            y = self.pos_y + 1
            while y < len(self.map):
                if (self.map[y][self.pos_x] >= 0):
                    raycast += 1
                else:
                    return raycast
                y += 1
            return raycast
        elif direction == 5:  # DOWN-LEFT
            y = self.pos_y + 1
            x = self.pos_x + 1
            while y >= 0 and x >= 0 and y < len(self.map) and x < len(self.map[self.pos_y]):
                if (self.map[y][x] >= 0):
                    raycast += 1
                else:
                    return raycast
                y += 1
                x -= 1
            return raycast
        elif direction == 6:  # LEFT
            x = self.pos_x - 1
            while x >= 0:
                if (self.map[self.pos_y][x] >= 0):
                    raycast += 1
                else:
                    return raycast
                x -= 1
            return raycast
        elif direction == 7:  # UP-LEFT
            y = self.pos_y - 1
            x = self.pos_x + 1
            while y >= 0 and x >= 0 and y < len(self.map) and x < len(self.map[self.pos_y]):
                if (self.map[y][x] >= 0):
                    raycast += 1
                else:
                    return raycast
                y -= 1
                x -= 1
            return raycast


class Game:
    map_food = 1
    map_empty = 0
    map_head = -1
    map_tail = -2
    movement_limit = 800
    reason_hit = 'REASON_HIT'
    reason_limit = 'REASON_MOVE_LIMIT'

    def __init__(self, size=16,snake_x=2,snake_y=2,show_UI=False, brain=None,cooldown=0.01,start_cooldown=0,controller=None,id=0,debug=False):
        self.size = size
        self.id=id
        self.debug=debug
        self.controller=controller
        self.moves=0
        self.reason=None
        self.show_UI=show_UI
        self.UI = None
        self.create_map()
        self.start_cooldown=start_cooldown
        self.snake = Snake(self, brain=brain,cooldown=cooldown,pos_x=snake_x,pos_y=snake_y)
        self.gameover = False
        self.food_x = 0
        self.food_y = 0
        self.spawn_food()
        self.brain = brain
        self.last_moves=[]
        # print(self.snake.sens_obstacle(2,wall=False))

    def get_score(self):
        return (self.snake.tail_size+self.moves+(-Game.movement_limit if (self.reason == Game.reason_limit and self.snake.tail_size<3) else 0))

    def create_map(self):
        self.map = []
        for y in range(self.size):
            lst = []
            for x in range(self.size):
                lst.append(0)
            self.map.append(lst)

    def game_over(self,reason):
        self.reason=reason
        if self.debug:
            print(f'Game Over R:{reason} S:{self.get_score()}')
        if self.controller!=None:
            self.controller.on_game_done(self)

    def reset_game(self):
        self.snake.tail.clear()
        self.snake.tail_size=0
        self.create_map()
        self.snake.set_head()
        self.snake.pos_y=2
        self.snake.pos_x=2
        self.movement_limit=0
        self.snake.set_head()
        self.UI.update_all()
        self.spawn_food()
        self.start_game()


    def make_move(self, old_pos, new_pos):
        self.gameover,reason = self.can_move(new_pos)
        self.gameover=not self.gameover
        if self.gameover:
            self.game_over(reason)
        return not self.gameover

    def spawn_food(self, food=map_food):
        while True:
            x = (random.randint(0, len(self.map[0]) - 1))
            y = (random.randint(0, len(self.map) - 1))
            if self.map[y][x] == Game.map_empty:
                self.map[y][x] = food
                self.food_x = x
                self.food_y = y
                self.update_ui(x,y)
                break

    def can_move(self, points):
        # print('can_move:',(points[0]<0 or points[1]<0),(points[0]>=self.size or points[1]>=self.size))
        # IF OUTSIDE THE GAME
        if points[0] < 0 or points[1] < 0 or points[0] >= self.size or points[1] >= self.size:
            return False,Game.reason_hit
        # IF IT IS FOOD OR EMPTY
        elif self.map[points[1]][points[0]] >= Game.map_empty:
            # IF IT IS FOOD
            if self.map[points[1]][points[0]] > Game.map_empty:
                self.snake.food_eaten()
                self.spawn_food()
            self.moves+=1
            # IF MOVEMENT REACHED
            if self.moves==Game.movement_limit:
                return False,Game.reason_limit
            return True,None
        # IF IT IS WALL
        else:
            return False,Game.reason_hit

    def print_map(self):
        for y in self.map:
            st = ''
            for x in y:
                st += '' + str(x) + '-'
            print(st)

    def set_ui(self, UI):
        self.UI = UI


    def update_ui(self,px,py):
        if not self.show_UI:
            return
        if self.UI != None:
            self.UI.single_update(px,py)
    def update_brain_ui(self):
        if not self.show_UI:
            return
        if self.UI != None:
            self.UI.update_brain()

    def start_game(self):
        time.sleep(self.start_cooldown)
        while not self.gameover:
            self.game_loop()
            self.update_brain_ui()

    def game_loop(self):
        self.snake.single_turn()



class SnakeUI(Frame):

    def __init__(self,game=None):
        super().__init__()
        self.color_input = '#ff550d'
        self.color_output = '#0d45ff'
        self.color_snake='#ffffff'
        self.color_snake_tail='#8a8a8a'
        self.color_wall='#025a7d'
        self.color_outline='#ffffff'
        self.color_background='#000000'
        self.color_food='#b32b32'

        self.game = game
        self.squares = []
        self.spheres = []
        self.initUI()
        #time.sleep(1)
        #self.status=self.canvas.create_text(600, 250, text="HELLO WORLD", fill="black", font=('Helvetica 15 bold'))
        self.canvas.pack()
        self.pack(fill=BOTH, expand=1)


        #threading.Thread(target=self.start_main_loop).start()

    def start_main_loop(self):
        self.root.mainloop()

    def set_game(self,game):
        self.squares = []
        self.spheres = []
        self.game=game
        if self.game != None:
            self.draw()
            self.draw_brain()

    def color_fade(self, color, percent):
        color = color.lstrip('#')
        lv = len(color)
        # color = np.array(color)
        color = tuple(int(color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
        white = np.array([255, 255, 255])
        vector = white - color
        color = (color + vector * percent)
        color=('#%02x%02x%02x' % (int(color[0]), int(color[1]), int(color[2]))).replace('-','')
        if len(color)>7:
            return color[:7]
        else:
            return color

    def find_color(self, index):
        if index == Game.map_empty:
            return (self.color_background, self.color_background)
        elif index == Game.map_head:
            return (self.color_snake, self.color_snake)
        elif index == Game.map_tail:
            return (self.color_snake_tail, self.color_snake_tail)
        elif index == Game.map_food:
            return (self.color_food, self.color_food)

    def draw(self):
        self.canvas.delete("all")
        size = len(self.game.map)
        rect_size_x = self.width / size
        rect_size_y = self.height / size
        for y in range(len(self.game.map)):
            lst = []
            for x in range(len(self.game.map[0])):
                color = self.find_color(self.game.map[y][x])
                lst.append(self.canvas.create_rectangle(x * rect_size_x, y * rect_size_y, (x + 1) * rect_size_x,
                                                        (y + 1) * rect_size_y, outline=self.color_background, fill=color[1]))
            self.squares.append(lst)
        self.canvas.pack(fill=BOTH, expand=1)

    def single_update(self,px,py):
        color = self.find_color(self.game.map[py][px])
        self.canvas.itemconfig(self.squares[py][px], outline=color[0], fill=color[1])
    def update_all(self):
        for y in range(len(self.game.map)):
            for x in range(len(self.game.map[y])):
                color = self.find_color(self.game.map[y][x])
                self.canvas.itemconfig(self.squares[y][x], outline=color[0], fill=color[1])

    def update_brain(self):
        if len(self.spheres) != 0:
            for index in range(7):
                self.canvas.itemconfig(self.spheres[0][index], outline='#000000',
                                       fill=self.color_fade(self.color_input, self.game.snake.brain.input[index] / self.game.size))
            for index in range(7):
                self.canvas.itemconfig(self.spheres[0][7+index], outline='#000000',
                                       fill=self.color_fade(self.color_input, 1-self.game.snake.brain.input[7+index]/self.game.size))
            self.canvas.itemconfig(self.spheres[0][14], outline='#000000',
                                   fill=self.color_fade(self.color_input,
                                                        1 - self.game.snake.brain.input[14]/self.game.size*4))
        if len(self.spheres) >= 2:
            last = len(self.spheres) - 1
            for index in range(len(self.spheres[last])):
                self.canvas.itemconfig(self.spheres[last][index], outline='#000000',
                                       fill=self.color_fade(self.color_output,
                                                            0 if (self.game.snake.brain.output + 1) == index else 1))

        if len(self.spheres)>=3:
            for m in range(len(self.game.snake.brain.midle)):
                for layer in range(len(self.game.snake.brain.midle[m])):
                    color = self.color_fade('#6d32a8',self.game.snake.brain.midle[m][layer])
                    self.canvas.itemconfig(self.spheres[1+m][layer], outline='#000000', fill=color)




    def draw_brain(self, width=300, height=200, start_from_x=550, start_from_y=10):

        shape = self.game.snake.brain.pre_draw()
        shape_size = len(shape)
        size_x = width / (shape_size + 1)
        # FINDING POINTS TO DRAW
        points = []
        for ci in range(len(shape)):
            lst = []
            size_y = height / (shape[ci] + 1)
            for cy in range(shape[ci]):
                lst.append((start_from_x + size_x * (1 + ci), start_from_y + size_y * (1 + cy)))
                # self.create_circle(start_from_x+size_x*(1+ci),start_from_y+size_y*(1+cy),5)
            points.append(lst)
        # LINES Color:"#9e9e9e"
        for x in range(0, len(points) - 1):
            start_lst = points[x]
            end_lst = points[x + 1]
            for first_point in start_lst:
                for second_point in end_lst:
                    self.canvas.create_line(first_point[0], first_point[1], second_point[0], second_point[1],
                                            fill=self.color_fade('#000000',0.7), width=1.2)
        # CREATING CIRCLES FOR NODE
        for colum in points:
            lst = []
            for node in colum:
                lst.append(self.create_circle(node[0], node[1], 5))
            self.spheres.append(lst)

    def create_circle(self, x, y, r, fill='#ffffff'):  # center coordinates, radius
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        return self.canvas.create_oval(x0, y0, x1, y1, fill=fill, outline='#000000')

    def initUI(self, width=500, height=500):
        self.width = width
        self.height = height
        self.master.title("Snake")
        self.canvas = Canvas(self)
        if self.game != None:
            self.draw()
            self.draw_brain()


class MainController:

    def __init__(self,games_count=10,map_size=30,generation_limit=30):
        self.generation=0
        self.generation_limit=generation_limit
        self.map_size=map_size
        self.game_count=games_count
        self.count_game_done=0
        self.games=[]
        self.lock = threading.Lock()
        self.best_brain = None
        self.UI=SnakeUI()
        #self.start_thread()

    def find_high_score(self):
        max_score=0
        best_brain=None
        for g in self.games:
            if max_score<g.get_score():
                max_score=g.get_score()
                best_brain=g.snake.brain
            #max_score=max(max_score,g.get_score())
        return (max_score,best_brain)

    def start_thread(self):
        threading.Thread(target=self.start_generation).start()
    def start_generation(self):
        #if self.count_game_done != self.game_count and self.count_game_done == 0:
        #    return
        self.count_game_done=0
        self.generation+=1
        score,brain=self.find_high_score()
        print(f'[New Gen] Generation:{self.generation}, ',f'best score:{score}' if self.generation!=0 else '')


        if brain != None:
            print(f'[BRAIN-UI] starting game with visual')
            g=Game(self.map_size,brain=brain,id=-1,show_UI=True,debug=True)
            self.UI.set_game(g)
            g.set_ui(self.UI)
            threading.Thread(target=g.start_game).start()

        if self.generation == self.generation_limit:
            return

        self.games.clear()
        for i in range(self.game_count):
            g = Game(self.map_size, show_UI=False, brain=SimpleBrain(), cooldown=0.001, start_cooldown=1, controller=self,
                 id=i + 1)
            self.games.append(g)
            threading.Thread(target=g.start_game).start()
            #print(f'[Game-Start] started game:{i+1}')




    def on_game_done(self,game:Game):
        with self.lock:
            self.count_game_done+=1
            print(f'[GameDone] Game-id: {game.id}, Game-Score: {game.get_score()}, Game-reason: {game.reason}')
            #self.UI.update_status(self.count_game_done,self.game_count)
            if self.count_game_done == self.game_count:
                self.start_thread()



def main():
    #mc = MainController()
    #mc.start_thread()
    #g = Game(30,show_UI=True,brain=SimpleBrain(),cooldown=0.001,start_cooldown=1)
    #g1 = Game(30,show_UI=True,brain=SimpleBrain(),cooldown=0.001,start_cooldown=1)
    #g2 = Game(30,show_UI=True,brain=SimpleBrain(),cooldown=0.001,start_cooldown=1)
    #g3 = Game(30,show_UI=True,brain=SimpleBrain(),cooldown=0.001,start_cooldown=1)

    root = Tk()

    #ex = SnakeUI()
    #ex.set_game(g)
    #g.set_ui(ex)

    mc = MainController(games_count=10,generation_limit=5)
    mc.start_thread()


    root.geometry("950x500+300+300")
    # multiprocessing.Process(target=g.start_game,args=(q,)).start()
    #threading.Thread(target=g.start_game).start()
    root.mainloop()
    #ex.root.mainloop()
    #threading.Thread(target=g1.start_game).start()
    #threading.Thread(target=g2.start_game).start()
    #threading.Thread(target=g3.start_game).start()


    # multiprocessing.Process(target= root.mainloop).start()
    #g.start_game()


if __name__ == '__main__':
    main()