from time import time
from math import sqrt
from dataclasses import dataclass
@dataclass
class state:
    power: int = 0
    is_serviceability: bool = False
    error_code = None


class Agent:
    def __init__(self, start_x: int, start_y: int, type, state: state, speed):
        self.x = start_x
        self.y = start_y
        self.type = type
        self.state = state
        self.speed = speed
        self.t_0 = time()

    def time2move(self, trajectory:list):
        all_time = []
        for x, y in trajectory:
            s = int(sqrt((self.x - x)**2 + (self.y - y)**2))
            t = s/self.speed
            all_time.append(t)
        return all_time

    def get_coordinate(self):
        return self.x, self.y


A1 = Agent(1, 2, None, None, 2)
x, y = A1.get_coordinate()
print(x, y)