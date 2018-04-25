#!usr/bin/python3
import numpy as np
import time
import matplotlib.pyplot as plt
"""Implement use method from:
http://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html"""

class Hungarian:
    star = 1
    prime = 2
    
    def __init__(self):
        self._original_cost_matrix = None
        self._cost_matrix = None
        self._transpose = False
        self._rows = None
        self._cols = None
        self._k = None
        self._row_starred = None
        self._col_starred = None
        self._row_covered = None
        self._col_covered = None
        self._label = None
        self._loc_prime = None

    def _step0(self):
        """Create an nxm  matrix called the cost matrix 
        in which each element represents the cost of 
        assigning one of n workers to one of m jobs.  
        Rotate the matrix so that there are at least 
        as many columns as rows and let k=min(n,m)."""
        m, n = np.shape(self._cost_matrix)
        if m > n:
            self._transpose = True
            self._cost_matrix = self._cost_matrix.transpose()
        else:
            self._transpose = False
        self._k = np.min([m, n])
        self._rows, self._cols = np.shape(self._cost_matrix)
        self._row_starred = np.zeros((self._rows, ), dtype=np.bool)
        self._col_starred = np.zeros((self._cols, ), dtype=np.bool)
        self._row_covered = np.zeros((self._rows, ), dtype=np.bool)
        self._col_covered = np.zeros((self._cols, ), dtype=np.bool)
        self._label = np.zeros_like(self._cost_matrix)

    def _step1(self):
        """For each row of the matrix, find the smallest 
        element and subtract it from every element in its 
        row.  Go to Step 2."""
        # TODO: Do not subtract smallest element both row and columns,
        # when matrix row number smaller than col number.
        self._cost_matrix -= np.min(self._cost_matrix, axis=1, keepdims=True)
    
    def _step2(self):
         """Find a zero (Z) in the resulting matrix.  
         If there is no starred zero in its row or column, star Z. 
         Repeat for each element in the matrix. Go to Step 3."""
         for row in range(self._rows):
            for col in range(self._cols):
                number = self._cost_matrix[row, col]
                if self._row_starred[row]==True or self._col_starred[col]==True or number!=0:
                    continue
                else:
                    self._label[row, col] = self.star
                    self._row_starred[row] = True
                    self._col_starred[col] = True
         
    def _step3(self):
        """Cover each column containing a starred zero.  
        If K columns are covered, the starred zeros describe a 
        complete set of unique assignments.  
        In this case, Go to DONE, otherwise, Go to Step 4."""
        np.copyto(self._col_covered, self._col_starred)
        if np.sum(self._col_covered) == self._k:
            return True
        else:
            return False
        
    def _step4(self):
        """Find a noncovered zero and prime it.  
        If there is no starred zero in the row containing 
        this primed zero, Go to Step 5.  
        Otherwise, cover this row and uncover the column 
        containing the starred zero. Continue in this 
        manner until there are no uncovered zeros left. 
        Save the smallest uncovered value and Go to Step 6.
        For example, the possible situations are, that there is a 
        noncovered zero which get primed and if there is no starred 
        zero in its row the program goes onto Step 5.  
        The other possible way out of Step 4 is that there are 
        no noncovered zeros at all, 
        in which case the program goes to Step 6."""
        # TODO: go to step 6 to find the smallest uncovered value
        while True:
            loc = self._find_uncoverd_zeros()
            if loc is None:
                return False
            row, col = loc
            self._label[row, col] = self.prime
            if self._row_starred[row] == False:
                self._loc_prime = (row, col)
                return True
            else:
                self._row_covered[row] = True
                self._col_covered[self._find_starred_in_row(row)] = False
       
    def _step5(self):
        """Construct a series of alternating primed and starred 
        zeros as follows.  Let Z0 represent the uncovered primed 
        zero found in Step 4.  Let Z1 denote the starred zero 
        in the column of Z0 (if any). Let Z2 denote the 
        primed zero in the row of Z1 (there will always be one).  
        Continue until the series terminates at a primed zero 
        that has no starred zero in its column.  
        Unstar each starred zero of the series, star each primed 
        zero of the series, erase all primes and uncover every 
        line in the matrix.  Return to Step 3."""
        row, col = self._loc_prime
        series = []
        series.append((row, col))
        while True:
            row = self._find_starred_in_col(col)
            if row != None:    
                series.append((row, col))
                col = self._find_primed_in_row(row)
                series.append((row, col))
            else:
                break
        for row, col in series:
            if self._label[row, col] == self.star:
                self._label[row, col] = 0
            elif self._label[row, col] == self.prime:
                self._label[row, col] = self.star
        for row, col in series:
            if self._label[row, col] == self.star:
                self._row_starred[row] = True
                self._col_starred[col] = True
        self._col_covered[:] = 0
        self._row_covered[:] = 0
        self._label[self._label==self.prime] = 0
        
    def _step6(self):
        """Add the value found in Step 4 to every element of each 
        covered row, and subtract it from every element of each 
        uncovered column.  Return to Step 4 without altering any stars, 
        primes, or covered lines."""    
        smallest_value = self._find_smallest_uncorverd_value()
        
        for row in range(self._rows):
            if self._row_covered[row] == True:
                self._cost_matrix[row, :] += smallest_value
        for col in range(self._cols):
            if self._col_covered[col] == False:
                self._cost_matrix[:, col] -= smallest_value
                
    def _find_uncoverd_zeros(self):
        for row in range(self._rows):
            for col in range(self._cols):
                number = self._cost_matrix[row, col]
                if self._col_covered[col]==True or self._row_covered[row]==True or number!=0:
                    continue
                else:
                    return (row, col)
        return None

    def _find_primed_in_row(self, row):
        for col in range(self._cols):
            if self._label[row, col] == self.prime:
                return col
        return None
        
    def _find_starred_in_col(self, col):
        for row in range(self._rows):
            if self._label[row, col] == self.star:
                return row 
        return None

    def _find_starred_in_row(self, row):
        for col in range(self._cols):
            if self._label[row, col] == self.star:
                return col
        return None

    def _find_smallest_uncorverd_value(self):
        smallest_uncoverd_value = None
        for row in range(self._rows):
            for col in range(self._cols):
                if self._col_covered[col]==True or self._row_covered[row]==True:
                    continue
                if smallest_uncoverd_value == None:
                    smallest_uncoverd_value = self._cost_matrix[row, col]
                elif self._cost_matrix[row, col] < smallest_uncoverd_value:
                    smallest_uncoverd_value = self._cost_matrix[row, col]
        return smallest_uncoverd_value         
    
    def solve(self, cost_matrix=np.random.randint(30, size=(5, 3))):
        self.__init__()
        self._original_cost_matrix = cost_matrix
        self._cost_matrix = cost_matrix.copy()
        self._step0()
        self._step1()
        self._step2()
        while self._step3() != True:
            while True:
                if self._step4() == True:
                    self._step5()
                    break
                else:            
                    self._step6()

        optimal_loc = []
        unassigned = []
        for col in range(self._label.shape[1]):
            col_has = False
            for row in range(self._label.shape[0]):
                if self._label[row, col] == self.star:
                    optimal_loc.append((row, col))
                    col_has = True
            if col_has == False:
                unassigned.append(col)
        if self._transpose:
            optimal_loc = [(y, x) for (x, y) in optimal_loc]

        if cost_matrix.shape[0] > cost_matrix.shape[1]:
            unassigned_row = unassigned
            unassigned_col = []
        elif cost_matrix.shape[0] < cost_matrix.shape[1]:
            unassigned_row = []
            unassigned_col = unassigned
        elif cost_matrix.shape[0] == cost_matrix.shape[1]:
            unassigned_row, unassigned_col = [], []
        return optimal_loc, unassigned_row, unassigned_col

# #!usr/bin/python3
# import numpy as np
# import time
# import matplotlib.pyplot as plt
# """Implement use method from:
# http://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html"""

# class Hungarian:
#     star = 1
#     prime = 2
    
#     def __init__(self):
#         self._original_cost_matrix = None
#         self._cost_matrix = None
#         self._transpose = False
#         self._rows = None
#         self._cols = None
#         self._k = None
#         self._row_starred = None
#         self._col_starred = None
#         self._row_covered = None
#         self._col_covered = None
#         self._label = None
#         self._loc_prime = None

#     def _step0(self):
#         """Create an nxm  matrix called the cost matrix 
#         in which each element represents the cost of 
#         assigning one of n workers to one of m jobs.  
#         Rotate the matrix so that there are at least 
#         as many columns as rows and let k=min(n,m)."""
#         m, n = np.shape(self._cost_matrix)
#         if m > n:
#             self._transpose = True
#             self._cost_matrix = self._cost_matrix.transpose()
#         else:
#             self._transpose = False
#         self._k = np.min([m, n])
#         self._rows, self._cols = np.shape(self._cost_matrix)
#         self._row_starred = np.zeros((self._rows, ), dtype=np.bool)
#         self._col_starred = np.zeros((self._cols, ), dtype=np.bool)
#         self._row_covered = np.zeros((self._rows, ), dtype=np.bool)
#         self._col_covered = np.zeros((self._cols, ), dtype=np.bool)
#         self._label = np.zeros_like(self._cost_matrix)

#     def _step1(self):
#         """For each row of the matrix, find the smallest 
#         element and subtract it from every element in its 
#         row.  Go to Step 2."""
#         # TODO: Do not subtract smallest element both row and columns,
#         # when matrix row number smaller than col number.
#         self._cost_matrix -= np.min(self._cost_matrix, axis=1, keepdims=True)
    
#     def _step2(self):
#         """Find a zero (Z) in the resulting matrix.  
#         If there is no starred zero in its row or column, star Z. 
#         Repeat for each element in the matrix. Go to Step 3."""
#         for row in range(self._rows):
#             for col in range(self._cols):
#                 number = self._cost_matrix[row, col]
#                 if self._row_starred[row] is True or self._col_starred[col] is True or number!=0:
#                     continue
#                 else:
#                     self._label[row, col] = self.star
#                     self._row_starred[row] = True
#                     self._col_starred[col] = True
         
#     def _step3(self):
#         """Cover each column containing a starred zero.  
#         If K columns are covered, the starred zeros describe a 
#         complete set of unique assignments.  
#         In this case, Go to DONE, otherwise, Go to Step 4."""
#         np.copyto(self._col_covered, self._col_starred)
#         if np.sum(self._col_covered) == self._k:
#             return True
#         else:
#             return False
        
#     def _step4(self):
#         """Find a noncovered zero and prime it.  
#         If there is no starred zero in the row containing 
#         this primed zero, Go to Step 5.  
#         Otherwise, cover this row and uncover the column 
#         containing the starred zero. Continue in this 
#         manner until there are no uncovered zeros left. 
#         Save the smallest uncovered value and Go to Step 6.
#         For example, the possible situations are, that there is a 
#         noncovered zero which get primed and if there is no starred 
#         zero in its row the program goes onto Step 5.  
#         The other possible way out of Step 4 is that there are 
#         no noncovered zeros at all, 
#         in which case the program goes to Step 6."""
#         # TODO: go to step 6 to find the smallest uncovered value
#         while True:
#             loc = self._find_uncoverd_zeros()
#             if loc is None:
#                 return False
#             row, col = loc
#             self._label[row, col] = self.prime
#             if self._row_starred[row] is False:
#                 self._loc_prime = (row, col)
#                 return True
#             else:
#                 self._row_covered[row] = True
#                 self._col_covered[self._find_starred_in_row(row)] = False
       
#     def _step5(self):
#         """Construct a series of alternating primed and starred 
#         zeros as follows.  Let Z0 represent the uncovered primed 
#         zero found in Step 4.  Let Z1 denote the starred zero 
#         in the column of Z0 (if any). Let Z2 denote the 
#         primed zero in the row of Z1 (there will always be one).  
#         Continue until the series terminates at a primed zero 
#         that has no starred zero in its column.  
#         Unstar each starred zero of the series, star each primed 
#         zero of the series, erase all primes and uncover every 
#         line in the matrix.  Return to Step 3."""
#         row, col = self._loc_prime
#         series = []
#         series.append((row, col))
#         while True:
#             row = self._find_starred_in_col(col)
#             if row != None:    
#                 series.append((row, col))
#                 col = self._find_primed_in_row(row)
#                 series.append((row, col))
#             else:
#                 break
#         for row, col in series:
#             if self._label[row, col] == self.star:
#                 self._label[row, col] = 0
#             elif self._label[row, col] == self.prime:
#                 self._label[row, col] = self.star
#         for row, col in series:
#             if self._label[row, col] == self.star:
#                 self._row_starred[row] = True
#                 self._col_starred[col] = True
#         self._col_covered[:] = 0
#         self._row_covered[:] = 0
#         self._label[self._label==self.prime] = 0
        
#     def _step6(self):
#         """Add the value found in Step 4 to every element of each 
#         covered row, and subtract it from every element of each 
#         uncovered column.  Return to Step 4 without altering any stars, 
#         primes, or covered lines."""    
#         smallest_value = self._find_smallest_uncorverd_value()
        
#         for row in range(self._rows):
#             if self._row_covered[row] is True:
#                 print(self._cost_matrix, smallest_value)
#                 self._cost_matrix[row, :] += smallest_value
#         for col in range(self._cols):
#             if self._col_covered[col] is False:
#                 self._cost_matrix[:, col] -= smallest_value
                
#     def _find_uncoverd_zeros(self):
#         for row in range(self._rows):
#             for col in range(self._cols):
#                 number = self._cost_matrix[row, col]
#                 if self._col_covered[col] is True or self._row_covered[row] is True or number!=0:
#                     continue
#                 else:
#                     return (row, col)
#         return None

#     def _find_primed_in_row(self, row):
#         for col in range(self._cols):
#             if self._label[row, col] == self.prime:
#                 return col
#         return None
        
#     def _find_starred_in_col(self, col):
#         for row in range(self._rows):
#             if self._label[row, col] == self.star:
#                 return row 
#         return None

#     def _find_starred_in_row(self, row):
#         for col in range(self._cols):
#             if self._label[row, col] == self.star:
#                 return col
#         return None

#     def _find_smallest_uncorverd_value(self):
#         smallest_uncoverd_value = None
#         for row in range(self._rows):
#             for col in range(self._cols):
#                 if self._col_covered[col] is True or self._row_covered[row] is True:
#                     continue
#                 if smallest_uncoverd_value is None:
#                     smallest_uncoverd_value = self._cost_matrix[row, col]
#                 elif self._cost_matrix[row, col] < smallest_uncoverd_value:
#                     smallest_uncoverd_value = self._cost_matrix[row, col]
#         return smallest_uncoverd_value         
    
#     def solve(self, cost_matrix=np.random.randint(30, size=(5, 3))):
#         self.__init__()
#         self._original_cost_matrix = cost_matrix
#         self._cost_matrix = cost_matrix.copy()
#         self._step0()
#         self._step1()
#         self._step2()
#         while self._step3() is not True:
#             while True:
#                 if self._step4() is True:
#                     self._step5()
#                     break
#                 else:            
#                     self._step6()

#         optimal_loc = []
#         unassigned = []
#         for col in range(self._label.shape[1]):
#             col_has = False
#             for row in range(self._label.shape[0]):
#                 if self._label[row, col] == self.star:
#                     optimal_loc.append((row, col))
#                     col_has = True
#             if col_has is False:
#                 unassigned.append(col)
#         if self._transpose:
#             optimal_loc = [(y, x) for (x, y) in optimal_loc]

        # if cost_matrix.shape[0] > cost_matrix.shape[1]:
        #     unassigned_row = unassigned
        #     unassigned_col = []
        # elif cost_matrix.shape[0] < cost_matrix.shape[1]:
        #     unassigned_row = unassigned
        #     unassigned_col = []
        # elif cost_matrix.shape[0] == cost_matrix.shape[1]:
        #     unassigned_row, unassigned_col = [], []
        # return optimal_loc, unassigned_row, unassigned_col


def main():
    solver = Hungarian()

    answer = solver.solve(np.array([[1, 2], [2, 5], [5, 6]]))
    print(answer)
    
    
def test():
    # ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ns = [10]
    spend_time = []
    for n in ns:
        seed = 10
        np.random.seed(seed)
        test_matrix = np.random.randn(n, n)
        
        solver = Hungarian()
        start = time.clock()
        answer = solver.solve(test_matrix)
        cost_time = time.clock() - start
        spend_time.append(cost_time)
        print('%d dims matrix cost time: %f' %(n, cost_time))
        
    plt.plot(ns, spend_time)
    plt.xlabel('n')
    plt.ylabel('cost time: /s')
    plt.show()
    
if __name__ == '__main__':
    main()
    
__all__ = ['Hungarian']