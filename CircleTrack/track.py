"""
Track multiple object use kalman filter, reimplement use python.
https://www.mathworks.com/help/vision/examples/motion-based-multiple-object-tracking.html"""
import numpy as np
import matplotlib.pyplot as plt

from tools import kalman as kf
from tools import assignment as assign
from tools import particlelist as ps

delt_t = 0.04
max_lose_time = 0.4

def distance(t1=np.zeros((1, 2)), t2=np.zeros((1, 2))):
    return np.sqrt(np.sum((t1-t2)**2))

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
    def __init__(self, x=np.array([[0], [0], [0], [0]])):
        super().__init__(x=x)
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
        return True if self._lose_time >= max_lose_time else False

# class Trackers:
#     def __init__(self):
#         self._tracker_list = []

#     def predict(self):
#         output = []
#         for i in range(len(self._tracker_list)):
#             output.append(self._tracker_list[i].predict())
#         return np.array(output)

#     def measurement(self, all_measure):
#         for i in range(len(self._tracker_list)):
#             self._tracker_list[i].measurement(all_measure[i])
#         for tracker in self._tracker_list[:]:
#             if not tracker.is_live():
#                 self._tracker_list.remove(tracker)

#     def append(self, measurement):
#         tracker = KF_live(x=np.array([[measurement[0]], measurement[1], [0], [0]]))
#         self._tracker_list.append(tracker)

class Solver:
    def __init__(self):
        self._measurements = []
        self._assign_method = assign.Hungarian()

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
            self._unassign_pred = list(range(0, self._predicts))
            self._unassign_measure = []
            return
        cost_matrix = distance_cost(np.array(self._predicts), np.array(self._measurements))
        max_cost = np.max(cost_matrix)
        cost_matrix_minimum = max_cost - cost_matrix
        
        self._optimal_loc, self._unassign_pred, self._unassign_measure = self._assign_method.solve(cost_matrix_minimum)

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
        if len(self._predicts) < len(self._measurements):
            for i in self._unassign_measure:
                kf_x = np.array([[self._measurements[i][0]], [self._measurements[i][1]], [0], [0]])
                k = KF_live(x=kf_x)
                self._trackers.append(k)

    def predict(self):
        # self._predicts = self._trackers.predict()
        self._predicts = []
        for i in range(len(self._trackers)):
            self._predicts.append(self._trackers[i].predict())
        return self._predicts

    def measurement(self, measurements):
        """measurements is a list (N*2)"""
        self._measurements = measurements
        self._assignment()
        self._correct()
        self._delete_losed_tracker()
        self._create_new_tracker()


def main():
    solver = Solver()

    n = 5
    measurements = ps.generate_measuremens(n)

    predicts = []
    for i in range(n):
        predict = solver.predict()
        solver.measurement(measurements[i])
        predicts.append(predict)    

    plt.figure()
    for measure in measurements:
        for x, y in measure:
            plt.plot(x, y, 'r.', markersize=1)

    plt.plot(3, 5, 'r.', markersize=8)
    plt.plot(8, 9, 'r^', markersize=8)
    plt.plot(10, 11, 'r+', markersize=8)
    plt.plot(1, 9, 'r*', markersize=8)
    plt.axis('off')
    # plt.show()

    # # plt.figure()
    # for pred in predicts:
    #     for point in pred:
    #         x, y = point[0, 0], point[1, 0]
    #         plt.plot(x, y, 'g.', markersize=1)
    # plt.title('5 objects')
    plt.show()


if __name__ == '__main__':
    main()
