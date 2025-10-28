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
    def __init__(self, start_x: int, start_y: int, type: bool, state: State, speed: int, alfa: int, color: str):
        self.x = start_x
        self.y = start_y
        self.type = type
        self.state = state
        self.speed = speed
        self.alfa = alfa
        # self.t_0 = time()
        self.color = color

    def get_coordinate(self):
        return self.x, self.y
    
    def up(self):
        pass
        

    def move(self):
        pass


class Board(Canvas):
    def __init__(self, cell_count: int, cell_size: int, line_size: int, cell_color: str, line_color: str):
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
        self.master.title("Робот")
        for i in range(cell_size+1):
            x = y = i*(self.cell_size + self.line_size)
            self.create_rectangle(0, y, self.board_size, y + self.line_size, outline=self.line_color, fill=self.line_color)
            self.create_rectangle(x, 0, self.line_size + x, self.board_size, outline=self.line_color, fill=self.line_color)
        self.pack()
    def add_agent(self, start_x: int, start_y: int, type: bool, state: State, speed: int, alfa: int, color: str):
        self.agents.append(Agent(start_x, start_y, type, state,  speed, alfa, color))
    def remove_agent(self, index):
        self.agents.pop(index)
    def run(self):
        pass


def main():
    root = Tk()
    cell_count = 10
    max_speed = 3
    alfs = [0, 90, 180, 270]
    board = Board(cell_count,20,2, "yellow","black")
    sp_color = ["red", "blue", "darkgreen", "brown", "purple", "coral"]
    for i in range(6):
        board.add_agent(random.randint(1, cell_count), random.randint(1, cell_count), None, None, random.randint(1, max_speed), alfs[random.randint(0, 3)], sp_color[i])
    root.mainloop()


if __name__ == '__main__':
    main()
