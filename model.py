from random import randint
import numpy as np


class ModelV1:

    def __init__(self, path):
        specs = self.read_file(path)
        self.shape = specs[0]
        self.row_constraints = list(reversed(specs[1:self.shape[1]+1]))
        self.column_constraints = list(reversed(specs[1+self.shape[1]:]))
        self.domains = [[[] for _ in range(self.shape[0])], [[] for _ in range(self.shape[1])]]
        self.x_segments = [[] for _ in range(self.shape[0])]
        self.y_segments = [[] for _ in range(self.shape[1])]
        self.generate_domains()

    def h(self):
        pass

    def generate_next(self, state):
        pass

    def generate_random_state(self):
        ret = []
        for domain in self.domains[0] + self.domains[1]:
            print(len(domain))
            ret.append(randint(0, len(domain)))
        return ret

    def generate_domains(self):
        for i, constraints in enumerate([self.column_constraints, self.row_constraints]):
            for j, variables in enumerate(constraints):
                k = 0
                while True:
                    l = k
                    d_set = []
                    for v in range(len(variables)):
                        d_set.append(l)
                        l += variables[v] + 1
                    k += 1
                    l -= 1
                    if l <= self.shape[1-i]:
                        self.domains[i][j].append(d_set)
                    else:
                        break

    def generate_segments(self, constraints, domain):
        pass


    def dimension_filtering(self):
        pass

    @staticmethod
    def read_file(path):
        with open(path, 'r') as f:
            ret = []
            for l in f:
                ret.append([int(x) for x in l.split(' ')])
        return ret
