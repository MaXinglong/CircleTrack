"""
Track multiple object use kalman filter, reimplement by python.
https://www.mathworks.com/help/vision/examples/motion-based-multiple-object-tracking.html"""
import datetime

import numpy as np
import matplotlib.pyplot as plt

from tools import kalman as kf
from tools import assignment as assign
from tools import particlelist as ps

delt_t = 0.04
max_lose_time = 0.1
min_distance_lose = 50
random_seed = 3


def distance(t1=np.zeros((1, 2)), t2=np.zeros((1, 2))):
    return np.sqrt(np.sum((t1.squeeze()-t2.squeeze())**2))

def distance_cost(worker=np.zeros((3, 2)), job=np.zeros((4, 2))):
    """calculate cost use distance.
    if the input is worker(3*2), job(4*2), the output
    dims will be (3*4)"""
    m, n = worker.shape[0], job.shape[0]
    output = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            output[i, j] = distance(worker[i, :], job[j, :])
    return output

class KF_live(kf.Kalman):
    def __init__(self, F=np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]), x=np.array([[0], [0], [0], [0]])):
        super().__init__(F=F, x=x)
        self._live_time = delt_t
        self._lose_time = 0

    def measurement(self, measure):
        if measure is None:
            measure = np.array([[self._x_pred[0, 0]], [self._x_pred[1, 0]]])
            super().measurement(measure)
            self._lose_time += delt_t
            self._live_time = 0
        else:
            super().measurement(measure)
            self._lose_time = 0
            self._live_time += delt_t
    
    def is_live(self):
        return False if self._lose_time >= max_lose_time else True

class assign_method(assign.Hungarian):
    def __init__(self, min_distance):
        super().__init__()
        self._min_distance = min_distance
    
    def solve(self, cost_matrix_input=np.random.randint(30, size=(5, 3))):
        cost_matrix = cost_matrix_input.copy()

        optimal_loc, unassigned_row, unassigned_col = super().solve(cost_matrix)
        
        for idx in optimal_loc:
            x, y = idx
##            if cost_matrix[x, y] > self._min_distance:
##                optimal_loc.remove(idx)
##                unassigned_row.append(x)
##                unassigned_col.append(y)
        return optimal_loc, unassigned_row, unassigned_col

class Solver:
    def __init__(self):
        self._measurements = []
        self._assign_method = assign_method(min_distance=min_distance_lose)

        self._optimal_loc = []
        self._unassign_measure = []
        self._unassign_pred = []
        
        self._trackers = []
        self._predicts = []

    def _assignment(self):
        if not self._predicts:
            self._optimal_loc = []
            self._unassign_pred = []
            self._unassign_measure = list(range(0, len(self._measurements)))
            return
        if not self._measurements:
            self._optimal_loc = []
            self._unassign_pred = list(range(0, len(self._predicts)))
            self._unassign_measure = []
            return
        cost_matrix = distance_cost(np.array(self._predicts), np.array(self._measurements))
        
        
        self._optimal_loc, self._unassign_pred, self._unassign_measure = self._assign_method.solve(cost_matrix)

    def _correct(self):
        """correct the kalman filter"""
        for i, j in self._optimal_loc:
            kf_x = np.array([[self._measurements[j][0]], [self._measurements[j][1]]])
            self._trackers[i].measurement(kf_x)
        for i in self._unassign_pred:
            self._trackers[i].measurement(None)

    def _delete_losed_tracker(self):
        for tracker in self._trackers[:]:
            if not tracker.is_live():
                self._trackers.remove(tracker)

    def _create_new_tracker(self):
        """create new tracker for the unassigned measurement"""
        for i in self._unassign_measure:
            kf_x = np.array([[self._measurements[i][0]], [self._measurements[i][1]], [0], [0]])
            k = KF_live(F=np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]), x=kf_x)
            self._trackers.append(k)

    def predict(self):
        self._predicts = []
        for i in range(len(self._trackers)):
            predict = self._trackers[i].predict()
            x, y = predict[0, 0], predict[1, 0]
            self._predicts.append([x, y])
        return self._predicts

    def measurement(self, measurements):
        """measurements is a list (N*2)"""
        self._measurements = measurements
        self._assignment()
        self._correct()
        self._delete_losed_tracker()
        self._create_new_tracker()


def generate_measuremens(sequence_num=1000):
    """generate measurements"""
    np.random.seed(random_seed)
    output = []
    particals = ps.ParticalList()
        
    for i in range(sequence_num):
        if np.random.random() < 0.005:
#        if i == 0:
            rand_x = np.random.random()*1000
            rand_y = np.random.random()*1000
            rand_xsp = max(np.random.random()*10, 5)*np.sign(np.random.randn())
            rand_ysp = max(np.random.random()*10, 5)*np.sign(np.random.randn())
            particals.append(init_x=rand_x, init_y=rand_y, x_sp=rand_xsp, y_sp=rand_ysp)

#        if particals.amounts() == 0:
#            rand_x = np.random.random()*1000
#            rand_y = np.random.random()*1000
#            rand_xsp = 10+np.random.random()*5
#            rand_ysp = 10+np.random.random()*5
#            particals.append(init_x=rand_x, init_y=rand_y, x_sp=rand_xsp, y_sp=rand_ysp)

        particals.update()
        measurement = particals.get_measurements()
        for idx in range(len(measurement)):
            measurement[idx][0] += np.random.randn()*10
            measurement[idx][1] += np.random.randn()*10
        output.append(measurement)
    return output


def extract_data(data):
    return [[x, y] for frame in data if len(frame)!=0 for x, y in frame]


def main():
    solver = Solver()

    n = 2000
    t1 = datetime.datetime.now()
    measurements = generate_measuremens(n)
    print(datetime.datetime.now() - t1)

    t1 = datetime.datetime.now()
    predicts = []
    for i in range(n):
        predict = solver.predict()
        solver.measurement(measurements[i])
        predicts.append(predict)
    print(datetime.datetime.now() - t1)

    plt.figure()
    data = np.array(extract_data(measurements))
    plt.plot(data[:, 0], data[:, 1], 'o', markersize=1)
    
    data = np.array(extract_data(predicts))
    plt.plot(data[:, 0], data[:, 1], 'o', markersize=1)
    plt.legend(['measurements', 'predicts'])
    plt.title('Track result')
    plt.show()


if __name__ == '__main__':
    main()

