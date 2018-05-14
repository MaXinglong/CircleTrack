"""This library consist of Partical and ParticalList.
All of the Particals lives in the ParticalList.
For our application, the ParticalList could control the 
Partical's amounts, when it appear
The ParticalList could calculate these several values at
time-step t: 1. amounts in ParticalList; 2. every Partical's
speed; 3. every Partical's location. 4. objects(with id in 
order of appear in time-step t and the distance with origin.
And the ParticalList should be convience for evaluating
MOT(Multiple Object Tracking) metrics.)
"""
import numpy as np

np.random.seed(123)

delt_t = 0.04
max_live_time = 30

class Partical:
    """A Partical"""
    def __init__(self, init_x=0, init_y=0, x_sp=0, y_sp=0,
                init_radius=100, max_live_time=30,
                mass=1):
        self._location = np.array([init_x, init_y])
        self._speed = np.array([x_sp, y_sp])
        self._r = init_radius
        self._live_time = 0
        self._max_live_time = max_live_time
        self._mass = mass
        self._force = np.array([0.0, 0.0])
        
    def set_live_time(self, i=0):
        self._live_time = i

    def update(self):
        ax, ay = self._force/self._mass
        self._force = np.array([0.0, 0.0])
        self._speed[0] += ax*delt_t
        self._speed[1] += ay*delt_t
        self._location += self._speed*delt_t*1
        self._live_time += delt_t

    def apply_force(self, force):
        self._force += force

    def is_dead(self):
        return self._live_time >= self._max_live_time

    def location(self):
        return self._location

    def speed(self):
        return self._speed

    def distance_with_origin(self):
        return np.sqrt(self._location[0]**2 + self._location[1]**2)


class ParticalList:
    """Save partical lists"""
    def __init__(self):
        self._list = []
    
    def append(self, init_x=0, init_y=0, 
                x_sp=0, y_sp=0,
                init_radius=100, max_live_time=30,
                mass=1):
        partical = Partical(init_x=init_x, init_y=init_y, x_sp=x_sp, y_sp=y_sp,
            init_radius=init_radius, max_live_time=max_live_time, mass=mass)
        self._list.append(partical)

    def update(self):
        for partical in self._list[:]:
            if partical.is_dead():
                self._list.remove(partical)
        for i in range(len(self._list)):
            speed = self._list[i].speed()
            norm = np.sqrt(np.sum(speed**2))
            norm = 1 if norm < 1e-3 else norm
            force = np.array([speed[1], -speed[0]])/norm*1
            self._list[i].apply_force(force)
            self._list[i].update()

    def get_measurements(self):
        measurements = []
        for i in range(len(self._list)):
            raw = self._list[i].location().copy() 
            measurements.append(raw)
        return measurements

    def amounts(self):
        return len(self._list)

    def info(self):
        info = []

def generate_measuremens(sequence_num=1000):
    """an example for generate measurements"""
    output = []
    particals = ParticalList()
        
    # rand_x = np.random.random()*1000
    # rand_y = np.random.random()*1000
    # rand_xsp = np.random.random()*10
    # rand_ysp = np.random.random()*10
    # particals.append(init_x=rand_x, init_y=rand_y, x_sp=rand_xsp, y_sp=rand_ysp)
    for i in range(sequence_num):
        if np.random.random() < 0.005:
            rand_x = np.random.random()*1000
            rand_y = np.random.random()*1000
            rand_xsp = 10
            rand_ysp = 10+np.random.random()*5
            particals.append(init_x=rand_x, init_y=rand_y, x_sp=rand_xsp, y_sp=rand_ysp)
        particals.update()
        measurement = particals.get_measurements()
        output.append(measurement.copy())
    # print(output)
    return output

__all__ = ['ParticalList']

