from tkinter import Tk, Canvas
import random
from time import time
from dataclasses import dataclass
import json


@dataclass


class State:
    power: int = 0
    is_serviceability: bool = False
    error_code = None #используемые значения: 'crush', 'out'


class Agent:
    def __init__(self, start_x: int, start_y: int, type_agent: str, state: State, speed: int, angle: int, color: str, name: str):
        self.x = start_x
        self.y = start_y
        self.type_agent = type_agent
        self.state = state
        self.speed = speed
        self.angle = angle
        # self.t_0 = time()
        self.color = color
        self.name = name

    def get_coordinate(self):
        return [self.x, self.y]
    
    def up(self):
        pass

    def move(self):
        if self.angle == 0:
            self.x += self.speed
        if self.angle == 90:
            self.y -= self.speed
        if self.angle == 180:
            self.x -= self.speed
        if self.angle == 270:
            self.y += self.speed


class Board(Canvas):
    def __init__(self, row_count: int, column_count: int, cell_size: int, line_size: int, cell_color: str, line_color: str, delay: int):
        self.row_count = row_count # количество ячеек по вертикали
        self.column_count = column_count # колмчество ячеек по горизонтали
        self.height = row_count * cell_size + line_size * (row_count+1)
        self.width = column_count * cell_size + line_size * (column_count+1)
        super().__init__(
            width=self.width, height=self.height,
            background=cell_color, highlightthickness=0
        )
        self.cell_size = cell_size
        self.line_color = line_color
        self.line_size = line_size
        self.cell_colour = cell_color
        self.agents = []
        self.delay = delay
        self.inGame = True
        self.master.title("Робот")
        self.lines = []
        for i in range(column_count+1):
            x = i*(self.cell_size + self.line_size) # шаг
            self.lines.append(self.create_rectangle(x, 0, self.line_size + x, self.height, outline=self.line_color, fill=self.line_color))
        for i in range(row_count+1):
            y = i * (self.cell_size + self.line_size) # шаг
            self.lines.append(self.create_rectangle(0, y, self.width, y + self.line_size, outline=self.line_color, fill=self.line_color))
        self.pack()

    def add_agent(self, start_x: int, start_y: int, type_agent: str, state: State, speed: int, angle: int, color: str, name: str):
        self.agents.append(Agent(start_x, start_y, type_agent, state,  speed, angle, color, name))

    def remove_agent(self, index):
        self.agents.pop(index)
    def tipper(self, delta, x, y, angle, color, name):
        points = []
        if angle == 0:
            points.append([delta * (x - 1), delta * y, delta * (x + 0.5), delta * (y + 1)])
            points.append([delta * (x + 0.5), delta * (y + 0.25), delta * (x + 1), delta * (y + 0.75)])
        if angle == 90:
            points.append([delta * x, delta * (y + 2), delta * (x + 1), delta * (y + 0.5)])
            points.append([delta * (x + 0.25), delta * (y + 0.5), delta * (x + 0.75), delta * y])
        if angle == 180:
            points.append([delta * (x + 2), delta * y, delta * (x + 0.5), delta * (y + 1)])
            points.append([delta * (x + 0.5), delta * (y + 0.25), delta * x, delta * (y + 0.75)])
        if angle == 270:
            points.append([delta * x, delta * (y - 1), delta * (x + 1), delta * (y + 0.5)])
            points.append([delta * (x + 0.25), delta * (y + 0.5), delta * (x + 0.75), delta * (y + 1)])
        self.create_rectangle(points[0][0], points[0][1], points[0][2], points[0][3], outline=color, fill=color, tag=name)
        self.create_oval(points[1][0], points[1][1], points[1][2], points[1][3], outline=color, fill=color, tag=name)

    def lift_truck(self, delta, x, y, angle):
        points = []
        if angle == 0:
            points = [delta * (x - 1), delta * y, delta * (x + 0.5), delta * y, delta * (x + 0.5), delta * (y + 0.25),
                      delta * (x + 1), delta * (y + 0.25), delta * (x + 1), delta * (y + 0.75), delta * (x + 0.5),
                      delta * (y + 0.75), delta * (x + 0.5), delta * (y + 1), delta * (x - 1), delta * (y + 1)]
        if angle == 90:
            points = [delta * x, delta * (y + 2), delta * x, delta * (y + 0.5), delta * (x + 0.25), delta * (y + 0.5),
                      delta * (x + 0.25), delta * y, delta * (x + 0.75), delta * y, delta * (x + 0.75),
                      delta * (y + 0.5), delta * (x + 1), delta * (y + 0.5), delta * (x + 1), delta * (y + 2)]
        if angle == 180:
            points = [delta * (x + 2), delta * y, delta * (x + 0.5), delta * y, delta * (x + 0.5), delta * (y + 0.25),
                      delta * x, delta * (y + 0.25), delta * x, delta * (y + 0.75), delta * (x + 0.5),
                      delta * (y + 0.75), delta * (x + 0.5), delta * (y + 1), delta * (x + 2), delta * (y + 1)]
        if angle == 270:
            points = [delta * x, delta * (y - 1), delta * x, delta * (y + 0.5), delta * (x + 0.25), delta * (y + 0.5),
                      delta * (x + 0.25), delta * (y + 1), delta * (x + 0.75), delta * (y + 1), delta * (x + 0.75),
                      delta * (y + 0.5), delta * (x + 1), delta * (y + 0.5), delta * (x + 1), delta * (y - 1)]
        return points

    def bulldozer(self, delta, x, y, angle):
        points = []
        if angle == 0:
            points = [delta * (x - 1), delta * y, delta * x, delta * y, delta * (x + 1), delta * (y + 0.5), delta * x,
                      delta * (y + 1), delta * (x - 1), delta * (y + 1)]
        if angle == 90:
            points = [delta * x, delta * (y + 2), delta * x, delta * (y + 1), delta * (x + 0.5), delta * y,
                      delta * (x + 1), delta * (y + 1), delta * (x + 1), delta * (y + 2)]
        if angle == 180:
            points = [delta * (x + 2), delta * y, delta * (x + 1), delta * y, delta * x, delta * (y + 0.5),
                      delta * (x + 1), delta * (y + 1), delta * (x + 2), delta * (y + 1)]
        if angle == 270:
            points = [delta * x, delta * (y - 1), delta * x, delta * y, delta * (x + 0.5), delta * (y + 1),
                      delta * (x + 1), delta * y, delta * (x + 1), delta * (y - 1)]
        return points

    def draw_agents(self):
        self.clear_agents()
        delta = self.cell_size + self.line_size # шаг
        for agent in self.agents:
            x = agent.get_coordinate()[0]
            y = agent.get_coordinate()[1]
            points = []
            if agent.type_agent == 'Tipper':
                self.tipper(delta, x, y, agent.angle, agent.color, agent.name)
            else:
                if agent.type_agent == 'Bulldozer':
                    points = self.bulldozer(delta, x, y, agent.angle)
                if agent.type_agent == 'Lift truck':
                    points = self.lift_truck(delta, x, y, agent.angle)
                self.create_polygon(points, outline=agent.color, fill=agent.color, tag=agent.name)

    def clear_agents(self):
        for agent in self.agents:
            self.delete(agent.name)

    def checkCollisions(self):
        for agent in self.agents:
            x = agent.get_coordinate()[0]
            y = agent.get_coordinate()[1]
            if (x < 0) or (x >= self.column_count) or (y < 0) or (y >= self.row_count):
                self.inGame = False
                agent.state.error_code = 'out'
            #agent_ui = self.find_withtag(agent.name)
            #x1, y1, x2, y2 = self.bbox(agent_ui)
            #overlap = self.find_overlapping(x1, y1, x2, y2)
            #for over in overlap:
                #if over != agent_ui and over not in self.lines:
                    #self.inGame = False
                    #agent.state.error_code = 'crush'

    def game_over(self):
        self.delete('all')
        self.create_text(self.width / 2, self.height / 2, font="Arial 32", fill='red', text="Game Over")

    def move_agents(self):
        for agent in self.agents:
            agent.move()

    def on_timer(self):
        self.move_agents()
        self.draw_agents()
        self.checkCollisions()
        if self.inGame:
            self.after(self.delay, self.on_timer)
        else:
            self.game_over()

    def run(self):
        self.draw_agents()
        self.on_timer()


def main():
    root = Tk()
    max_speed = 2
    coordinates = [[3, 2], [5, 2], [7, 2], [9, 2], [11, 2], [13, 2]]
    angles = [0, 90, 180, 270]
    board = Board(15, 15, 60,  2, "yellow", "black", 1000)
    max_x = board.column_count - 3
    max_y = board.row_count - 3
    types_agents = ['Bulldozer', 'Lift truck', 'Tipper']
    colors = ["red", "blue", "darkgreen", "brown", "purple", "coral"]
    for i in range(6):
        board.add_agent(random.randint(5,  11), random.randint(5, 11), types_agents[random.randint(0,2)], State,
                        random.randint(1, 1), angles[random.randint(0, 3)], colors[i], "agent" + str(i))
    board.run()
    root.mainloop()


if __name__ == '__main__':
    main()
