from time import time
from math import sqrt
from dataclasses import dataclass
import numpy as np
import json
@dataclass

class state:
    power: int = 0
    is_serviceability: bool = False
    error_code = None


class Agent:
    def __init__(self, start_x: int, start_y: int, type, state: state, speed, alfa):
        self.x = start_x
        self.y = start_y
        self.type = type
        self.state = state
        self.speed = speed
        self.alfa = alfa
        # self.t_0 = time()

    def get_coordinate(self):
        return self.x, self.y
    
    def up(self):
        pass
        

    def move(self, ):
        pass

class matrix:
    def __init__(self, size: int):
        self.size = size
        self.agents = []
    def add_agent(self, start_x: int, start_y: int, type, state: state, speed, alfa):
        self.agents.append(Agent(start_x, start_y, type, state,  speed, alfa))
    def remove_agent(self, index):
        self.agents.pop(index)
    def run(self):
        pass


