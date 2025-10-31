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
    
matrix_1 = np.zeros ((10, 10))
matrix_2 = np.ones((2, 3))
s=matrix_1 [7 : 10, : 3]
matrix=matrix_1+matrix_2
print(matrix)
        
    


A1 = Agent(1, 2, None, None, 2)
x, y = A1.get_coordinate()
print(x, y)