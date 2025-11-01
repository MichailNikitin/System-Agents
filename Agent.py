from tkinter import Tk, Frame, Canvas
import random
from time import time
from dataclasses import dataclass
import numpy as np
import json
@dataclass


class State:
    power: int = 0
    is_serviceability: bool = False
    error_code = None


class Agent:
    def __init__(self, start_x: int, start_y: int, type: bool, state: State, speed: int, alfa: int, color: str, name: str):
        self.x = start_x
        self.y = start_y
        self.type = type
        self.state = state
        self.speed = speed
        self.alfa = alfa
        # self.t_0 = time()
        self.color = color
        self.name = name

    def get_coordinate(self):
        return [self.x, self.y]
    
    def up(self):
        pass

    def move(self):
        if self.alfa == 0:
            self.x += self.speed
        if self.alfa == 90:
            self.y -= self.speed
        if self.alfa == 180:
            self.x -= self.speed
        if self.alfa == 270:
            self.y += self.speed



class Board(Canvas):
    def __init__(self, cell_count: int, cell_size: int, line_size: int, cell_color: str, line_color: str, delay: int):
        self.board_size = cell_count * cell_size + line_size * (cell_count+1)
        super().__init__(
            width=self.board_size, height=self.board_size,
            background=cell_color, highlightthickness=0
        )
        self.cell_size = cell_size
        self.line_color = line_color
        self.line_size = line_size
        self.cell_colour = cell_color
        self.cell_count = cell_count
        self.agents = []
        self.delay = delay
        self.master.title("Полигон")
        for i in range(cell_size+1):
            x = y = i*(self.cell_size + self.line_size)
            self.create_rectangle(0, y, self.board_size, y + self.line_size, outline=self.line_color, fill=self.line_color)
            self.create_rectangle(x, 0, self.line_size + x, self.board_size, outline=self.line_color, fill=self.line_color)
        self.pack()

    def add_agent(self, start_x: int, start_y: int, type: bool, state: State, speed: int, alfa: int, color: str, name: str):
        self.agents.append(Agent(start_x, start_y, type, state,  speed, alfa, color, name))

    def remove_agent(self, index):
        self.agents.pop(index)

    def draw_agents(self):
        self.clear_agents()
        delta = self.cell_size + self.line_size
        for agent in self.agents:
            x = agent.get_coordinate()[0]
            y = agent.get_coordinate()[1]
            if agent.alfa == 0:
                points = [delta*(x-1), delta*y, delta*x, delta*y, delta*(x+1), delta*(y+0.5), delta*x, delta*(y+1), delta*(x-1), delta*(y+1)]
            if agent.alfa == 90:
                points = [delta*x, delta*(y+2), delta*x, delta*(y+1), delta*(x+0.5), delta*y, delta*(x+1), delta*(y+1), delta*(x+1), delta*(y+2)]
            if agent.alfa == 180:
                points = [delta*(x+2), delta*y, delta*(x+1), delta*y, delta*x, delta*(y + 0.5), delta*(x+1), delta*(y + 1), delta*(x+2), delta*(y+1)]
            if agent.alfa == 270:
                points = [delta * x, delta * (y - 1), delta * x, delta * y, delta * (x + 0.5), delta * (y + 1),
                          delta * (x + 1), delta * y, delta * (x + 1), delta * (y - 1)]
            self.create_polygon(points, outline=agent.color, fill=agent.color, tag=agent.name)

    def clear_agents(self):
        for agent in self.agents:
            self.delete(agent.name)


    def move_agents(self):
        for agent in self.agents:
            agent.move()

    def on_timer(self):
        self.draw_agents()
        self.move_agents()
        self.after(self.delay, self.on_timer)

    def run(self):
        self.draw_agents()
        self.on_timer()


def main():
    root = Tk()
    cell_count = 10
    max_speed = 3
    alfs = [0, 90, 180, 270]
    board = Board(cell_count, 50, 2, "yellow", "black", 2000)
    sp_color = ["red", "blue", "darkgreen", "brown", "purple", "coral"]
    for i in range(6):
        board.add_agent(random.randint(1, cell_count-2), random.randint(1, cell_count-2), None, None,
                        random.randint(1, max_speed), alfs[random.randint(0, 3)], sp_color[i], "agent" + str(i))
    board.run()
    root.mainloop()


if __name__ == '__main__':
    main()
