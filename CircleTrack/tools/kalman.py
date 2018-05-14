import numpy as np

class Kalman:
    def __init__(self, 
            F=np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]),
            P = np.array([[1000, 0, 0, 0], [0, 1000, 0, 0], [0, 0, 1000, 0], [0, 0, 0, 1000]]),
            Q = np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
            R = np.array([[10, 0], [0, 10]]),
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
            x = np.array([[0], [0], [0], [0]]),
            u = np.array([[0], [0], [0], [0]])):
        self._F = F
        self._P = P
        self._u = u
        self._Q = Q
        self._R = R
        self._H = H
        self._x = x

        self._x_pred = None

    def predict(self):
        self._x_pred = self._F.dot(self._x) + self._u
        self._P = self._F.dot(self._P).dot(self._F.transpose()) + self._Q
        return self._x_pred

    def measurement(self, measure):
        z = measure
        K = self._P.dot(self._H.transpose()).dot(np.linalg.inv(self._H.dot(self._P).dot(self._H.transpose()) + self._R))
        self._x = self._x_pred + K.dot(z - self._H.dot(self._x_pred))
        self._P = (np.eye(K.shape[0])-K.dot(self._H)).dot(self._P)
   

__all__ = ['Kalman']
