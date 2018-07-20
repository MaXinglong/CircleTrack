"""simulation"""
import numpy as np
import cv2

fps = 9.0
delt_time = 1.0/fps
partical_alive_min_area = 1 # TODO check

width = 500
height = 500

poisson_lamb = 2
# speed_kp, speed_ki, speed_kd = 10, 0,0
speed_k = 1
max_speed = 10
max_radius = 15
max_distance = np.sqrt(width**2+height**2)

def poisson(lamb, t=0):
    # return -lamb*t**2 + 2*t
    return np.power(lamb, t) / np.math.factorial(np.int(t)) * np.exp(-lamb)

def polym(t=0):
    return -2*t**2 + 10*t + partical_alive_min_area

def distance(point_1=0, point_2=0):
    return np.sqrt(point_1**2+point_2**2)

class Partical:
    def __init__(self, init_x=0, init_y=0, 
            target_x=0, target_y=0, init_radius=500):
        self.x, self.y = init_x, init_y
        self.area = 0 # TODO Radius
        self.init_radius=init_radius
        self.radius = 0
        self.alive_time = 0
        self.contour = None
        
        self.x_speed, self.y_speed = np.random.randn()*10, np.random.randn()*10
        self.x_acc, self.y_acc = 0, 0
        self.target_x, self.target_y = target_x, target_y

    @property
    def is_dead(self):
        # if self.area < partical_alive_min_area:
        if (self.radius<partical_alive_min_area) |\
                 (self.x<(0-max_radius)) | (self.x>(height+max_radius)) | \
                 (self.y<(0-max_radius)) | (self.y>(width+max_radius)):
            return True
        else:
            return False
    
    def _update_speed(self):
        self.x_acc = np.sign(self.target_x-self.x)*distance(self.x, self.target_x)/max_distance*speed_k
        self.y_acc = np.sign(self.target_y-self.y)*distance(self.y, self.target_y)/max_distance*speed_k
        self.x_speed += self.x_acc
        self.y_speed += self.y_acc
        if self.x_speed > max_speed: self.x_speed = max_speed
        if self.x_speed < -max_speed: self.x_speed = -max_speed
        if self.y_speed > max_speed: self.y_speed = max_speed
        if self.y_speed < -max_speed: self.y_speed = -max_speed

    def update_params(self):
        self._update_speed()

        self.x += self.x_speed
        self.y += self.y_speed
        self.area = 0
        self.radius = self._radius_of_live_lime(self.alive_time)
        self.alive_time += delt_time

    def _radius_of_live_lime(self, live_time):
        # return poisson(poisson_lamb, live_time)*self.init_radius
        tmp = polym(live_time)
        return tmp if tmp >= 0 else 0

class System:
    def __init__(self):
        self.partical_list = []
        self.run_time = 0

    @property
    def amounts(self):
        return len(self.partical_list)

    def update(self):
        self.run_time += delt_time
        for i in self.partical_list[:]:
            i.update_params()
            if i.is_dead:
                self.partical_list.remove(i)
    
    def add_partical(self, i):
        self.partical_list.append(i)

    def draw(self):
        pic_raw = np.zeros((height, width), dtype=np.uint8)
        for p in self.partical_list:
            cv2.circle(pic_raw, (np.int(p.y), np.int(p.x)), np.int(p.radius), 255, -1)
        return pic_raw

count = 0
def multi_partical():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_multi_long.avi',fourcc, fps, (width, height), False)

    np.random.seed(0)
    system = System()
    # while True:
    for w in range(0, 2000):
        print('partical amounts: ', system.amounts, 
            ' system runtime: %0.2f'%system.run_time)
        system.update()
        if np.random.randn(1) < 0.2:
            new_partical = Partical(
                init_x=np.random.randint(height), 
                init_y=np.random.randint(width),
                target_x=np.random.randint(height),
                target_y=np.random.randint(width),
                init_radius=np.random.randint(max_radius))
            system.add_partical(new_partical)
            global count
            count = count + 1
            print(count)
        pic = system.draw()
        cv2.imshow('system', pic)
        
        out.write(pic)

        if cv2.waitKey(np.int(delt_time*1000)) & 0xff == 27:
            break
        
    # out.release()

def single_partical():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_single.avi',fourcc, fps, (width, height), False)

    np.random.seed(0)
    system = System()

    new_partical = Partical(
        init_x=np.random.randint(height), 
        init_y=np.random.randint(width),
        target_x=np.random.randint(height),
        target_y=np.random.randint(width),
        init_radius=np.random.randint(max_radius))
    system.add_partical(new_partical)

    while True:
        system.update()

        pic = system.draw()
        cv2.imshow('system', pic)
        
        out.write(pic)

        if new_partical.is_dead:
            break
        if cv2.waitKey(np.int(delt_time*1000)) & 0xff == 27:
            break

def two_partical():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_two.avi',fourcc, fps, (width, height), False)

    np.random.seed(0)
    system = System()

    new_partical = Partical(
        init_x=np.random.randint(height), 
        init_y=np.random.randint(width),
        target_x=np.random.randint(height),
        target_y=np.random.randint(width),
        init_radius=np.random.randint(max_radius))
    system.add_partical(new_partical)
    new_partical = Partical(
        init_x=np.random.randint(height), 
        init_y=np.random.randint(width),
        target_x=np.random.randint(height),
        target_y=np.random.randint(width),
        init_radius=np.random.randint(max_radius))
    system.add_partical(new_partical)


    while True:
        system.update()

        pic = system.draw()
        cv2.imshow('system', pic)
        
        out.write(pic)

        if new_partical.is_dead:
            break
        if cv2.waitKey(np.int(delt_time*1000)) & 0xff == 27:
            break

if __name__ == '__main__':
    isMulti = 0
    if isMulti == 0:
        multi_partical()
    if isMulti == 1:
        single_partical()
    if isMulti == 2:
        two_partical()