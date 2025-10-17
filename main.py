import time
from dataclasses import dataclass
from math import sqrt
@dataclass
class state:
    error_code: None
    power: int = 0
    is_Serviceability: bool = False

class Agent:
    def __init__ (self, start_x, start_y, type, state: state, speed):
        self.x = start_x
        self.y = start_y
        self.type = type
        self.state = state
        self.speed = speed
        self.t_0 = time()

    def time2move(self, trajectory:list):
        all_time = []
        for x, y in trajectory:
            s = sqrt((self.x - x)**2 + (self.y - y)**2)
            t = s/self.speed
            all_time.append(t)
        return all_time

    def get_coordinate(self):
        return self.x, self.y

class Polegon:
    def __init__(self, zeroX, zeroY, endX, endY, zapret):
        self.zeroX = zeroX
        self.zeroY = zeroY
        self.endX = endX
        self.endY = endY
        self.zapret:list

    def fill_zapret_zone(self, zapret:list):



A1 = Agent(1,2, None,None,2)
x,y = A1.get_coordinate()
print(x,y)

