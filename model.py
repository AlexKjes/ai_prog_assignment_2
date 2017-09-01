from random import randint
import numpy as np
import tkinter as tk


class ModelV1:

    def __init__(self, path):
        specs = self.read_file(path)
        self.shape = specs[0]
        self.row_constraints = list(reversed(specs[1:self.shape[1]+1]))
        self.column_constraints = list(reversed(specs[1+self.shape[1]:]))
        self.domains = [[[] for _ in range(self.shape[0])], [[] for _ in range(self.shape[1])]]
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


class ModelV2:

    def __init__(self, path):
        specs = self.read_file(path)
        self.shape = specs[0]
        self.row_constraints = list(reversed(specs[1:self.shape[1]+1]))
        self.column_constraints = list(reversed(specs[1+self.shape[1]:]))
        self.segments = [[[] for _ in range(self.shape[0])], [[] for _ in range(self.shape[1])]]
        self.generate_domains()

        self.tk_master = None
        self.tk_canvas = None

    def h(self):
        pass

    def generate_next(self, state):
        pass

    def validate(self, state):
        segments = self.get_segments(state)
        for s in segments:
            print(s)
        print("----------")
        print(segments[0][:])

    def generate_random_state(self):
        return [randint(0, len(d)-1) for d in self.segments[1]]

    def generate_domains(self):
        for i, constraints in enumerate([self.column_constraints, self.row_constraints]):
            for j, variables in enumerate(constraints):
                k = 0
                while True:
                    l = k
                    segment = np.zeros((self.shape[1-i], 1))
                    for v in range(len(variables)):
                        for m in range(l, l+variables[v]):
                            segment[m][0] = 1
                        l += variables[v] + 1
                    k += 1
                    l -= 1
                    if l < self.shape[1-i]:
                        self.segments[i][j].append(segment)
                    else:
                        self.segments[i][j].append(segment)
                        break

    def generate_segments(self, constraints, domain):
        pass

    def dimension_filtering(self):
        pass

    def draw_state(self, state):
        scale = 50
        if self.tk_master is None:
            self.tk_master = tk.Tk()
            self.tk_canvas = tk.Canvas(self.tk_master, width=self.shape[0]*scale, height=self.shape[1]*scale)
            self.tk_canvas.pack()

        segments = self.get_segments(state)

        for y, s in enumerate(segments):
            for x, p in enumerate(s.T):
                if p[0] == 1:
                    self.tk_canvas.create_rectangle(x*scale, y*scale, (x+1)*scale, (y+1)*scale, fill='red')


        self.tk_master.update()


    def get_segments(self, state):
        ret = []
        for i, d in enumerate(state):
            ret.append(self.segments[1][i][d].T)
        return ret

    @staticmethod
    def read_file(path):
        with open(path, 'r') as f:
            ret = []
            for l in f:
                ret.append([int(x) for x in l.split(' ')])
        return ret
