import numpy as np
import time

"""Implement use method from:
http://www.hungarianalgorithm.com/"""

class Hungarian:
    def __init__(self, original_cost_matrix=np.random.randn(3,3)):
        self._original_cost_matrix = original_cost_matrix
        self._temp_matrix = original_cost_matrix.copy()
        self._optimal_loc = []
    
    def _make_matrix_square(self):
        m, n = np.shape(self._original_cost_matrix)
        max_dim = max(m, n)
        self._temp_matrix = np.zeros((max_dim, max_dim))
        self._temp_matrix[0:m, 0:n] = self._original_cost_matrix
    
    def _substract_row_minima(self):
        self._temp_matrix -= np.min(self._temp_matrix, axis=1, keepdims=True)
        
    def _substract_col_minima(self):
        self._temp_matrix -= np.min(self._temp_matrix, axis=0, keepdims=True)
        
    def _is_enough_lines_cover_zeros(self):
        """if the axis contained all numbers in the shape, it is enough"""
        temp = np.argwhere(self._temp_matrix==0)
        if len(list(set(list(temp[:, 0])))) < self._temp_matrix.shape[0]:
            return False
        if len(list(set(list(temp[:, 1])))) < self._temp_matrix.shape[1]:
            return False
        return True
    
    def _cover_all_zeros_with_minimum_numer_lines(self):
        paint = np.zeros_like(self._temp_matrix)
        paint[self._temp_matrix!=0] = 1
        label = np.zeros_like(self._temp_matrix)
        zero_loc = self._find_first_zero_loc(paint)
        optimal_loc = []
        while zero_loc is not None:
            row, col = zero_loc
            print(zero_loc)
            row_zeros_amount = np.sum(paint[row, :]==0)
            col_zeros_amount = np.sum(paint[:, col]==0)
            if row_zeros_amount >= col_zeros_amount:
                label[row, :] += 1
                paint[row, :] = 1
            else:
                label[:, col] += 1
                paint[:, col] = 1
            optimal_loc.append((row, col))
            zero_loc = self._find_first_zero_loc(paint)
        return label, optimal_loc

    def _find_first_zero_loc(self, paint):
        start_id = 0
        for idx in range(start_id, paint.shape[0]*paint.shape[1]):
            row, col = idx//paint.shape[1], idx%paint.shape[1]
            if paint[row, col] == 0:
                return (row, col)
        return None

    def _create_additional_zeros(self, label):
        smallest_uncoverd_number = np.min(self._temp_matrix[label==0])
        self._temp_matrix[label==0] -= smallest_uncoverd_number
        self._temp_matrix[label==2] += smallest_uncoverd_number
        
    def _optimal_value(self, optimal_loc):
        val = 0
        for row, col in optimal_loc:
            val += self._original_cost_matrix[row, col]
        return val
    
    def set_original_cost_matrix(self, original_cost_matrix=np.random.randn(3, 3)):
        self.__init__(original_cost_matrix)
    
    def solve(self):
        self._substract_row_minima()
        print(self._temp_matrix)
        self._substract_col_minima()
        print(self._temp_matrix)
        optimal_loc = []
        while len(optimal_loc)!=self._temp_matrix.shape[0]:
            label, optimal_loc = self._cover_all_zeros_with_minimum_numer_lines()
            self._create_additional_zeros(label)
        return optimal_loc, self._optimal_value(optimal_loc)
    
    
def main():
    test_matrix = np.array([[32, 97, 83, 95, 15, 88], [13, 32, 47, 92, 45, 30], [90, 17, 17, 43, 73, 9], [37, 67, 55, 72, 42, 36], [84, 69, 75, 59, 85, 53], [25, 22, 43, 93, 25, 87]])

    solver = Hungarian()
    solver.set_original_cost_matrix(test_matrix)
    print(solver.solve())
    
def factorial(n):
    return n*factorial(n-1) if n > 1 else 1

def C_m_n(m, n):
    return factorial(n)/(factorial(m)*factorial(n-m))

def A_m_n(m, n):
    return factorial(n)/factorial(n-m)

if __name__ == '__main__':
    main()
    
