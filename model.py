from random import randint
import numpy as np
import tkinter as tk
from csp_graph import Graph

class Model:

    def __init__(self, path):
        specs = self.read_file(path)
        self.csp_graph = Graph()
        self.shape = specs[0]
        self.row_hints = list(reversed(specs[1:self.shape[1]+1]))
        self.column_hints = list(reversed(specs[1+self.shape[1]:]))
        self.row_variables = [[] for _ in range(len(self.row_hints))]
        self.column_variables = [[] for _ in range(len(self.column_hints))]
        self.generate_domains_and_1d_constraints()
        self.generate_2d_constraints()

    def h(self):
        pass

    def generate_next(self, state):
        pass

    def generate_random_state(self):
        pass

    def generate_domains_and_1d_constraints(self):
        for i, row in enumerate(self.row_hints):
            for _ in row:
                self.row_variables[i].append(self.csp_graph.add_node([x for x in range(self.shape[0])]))
            for j, h in enumerate(row):
                if j == len(row)-1:  # if last variable
                    self.csp_graph.add_edge(self.row_variables[i][j], self.row_variables[i][j], lambda x, y: x+h-1 < self.shape[0])
                else:
                    self.csp_graph.add_edge(self.row_variables[i][j], self.row_variables[i][j+1], lambda x, y: x+h < max(y))
                if j != 0:  # if not first variable
                    self.csp_graph.add_edge(self.row_variables[i][j], self.row_variables[i][j-1], lambda x, y: x > min(y)+row[j-1])
        for i, column in enumerate(self.column_hints):
            for _ in column:
                self.column_variables[i].append(self.csp_graph.add_node([x for x in range(self.shape[1])]))
            for j, h in enumerate(column):
                if j == len(column)-1:  # if last variable
                    self.csp_graph.add_edge(self.column_variables[i][j], self.column_variables[i][j], lambda x, y: x+h-1 < self.shape[1])
                else:
                    self.csp_graph.add_edge(self.column_variables[i][j], self.column_variables[i][j+1], lambda x, y: x+h < max(y))
                if j != 0:  # if not first variable
                    self.csp_graph.add_edge(self.column_variables[i][j], self.column_variables[i][j-1], lambda x, y: x > min(y)+column[j-1])

    def generate_2d_constraints(self):
        for x, cvs in enumerate(self.column_variables):
            for y, rvs in enumerate(self.row_variables):
                for v in

    def generate_segments(self, constraints, domain):
        pass

    def dimension_filtering(self):
        pass

    def draw_state(self, state):
        scale = 50
        if self.tk_master is None:
            self.tk_master = tk.Tk()
            self.tk_canvas = tk.Canvas(self.tk_master, width=self.shape[0] * scale, height=self.shape[1] * scale)
            self.tk_canvas.pack()

        self.tk_canvas.delete('all')
        self.tk_canvas.create_rectangle(0, 0, self.shape[0] * scale, self.shape[1] * scale, fill='white')

        # TODO actually draw something

        self.tk_master.update()

    @staticmethod
    def read_file(path):
        with open(path, 'r') as f:
            ret = []
            for l in f:
                ret.append([int(x) for x in l.split(' ')])
        return ret



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
        for x in range(self.shape[0]):
            match = False
            for c in self.segments[0][x]:
                a = np.array(segments[:, x])
                b = np.array(c.T[0])
                if np.equal(a, b).all():
                    match = True
            if not match:
                return False

        return True


    def generate_random_state(self):
        return [randint(0, len(d)-1) for d in self.segments[1]]

    def generate_domains(self):
        for i, constraints in enumerate([self.column_constraints, self.row_constraints]):
            for j, variables in enumerate(constraints):
                k = 0
                while True:
                    l = k
                    segment = np.zeros((self.shape[1-i], 1), dtype=np.int8)
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

        self.tk_canvas.delete('all')
        self.tk_canvas.create_rectangle(0, 0, self.shape[0]*scale, self.shape[1]*scale, fill='white')

        segments = self.get_segments(state)
        for y, s in enumerate(segments):
            for x, p in enumerate(s):
                if p == 1:
                    self.tk_canvas.create_rectangle(x*scale, y*scale, (x+1)*scale, (y+1)*scale, fill='red')


        self.tk_master.update()


    def get_segments(self, state):
        ret = self.segments[1][0][state[0]].T
        for i, d in enumerate(state[1:]):
            ret = np.concatenate((ret, self.segments[1][i+1][d].T))
        return ret

    def shift_state(self, state, y):
        if state[y] + 1 < len(self.segments[1][y]):
            state[y] += 1
            return state
        else:
            state[y] = 0
            return state

    @staticmethod
    def read_file(path):
        with open(path, 'r') as f:
            ret = []
            for l in f:
                ret.append([int(x) for x in l.split(' ')])
        return ret


class ModelV3:

    def __init__(self, path):
        specs = self.read_file(path)
        self.shape = specs[0]
        self.row_hints = list(reversed(specs[1:self.shape[1]+1]))
        self.column_hints = list(reversed(specs[1+self.shape[1]:]))
        self.row_domains = [[] for _ in range(len(self.row_hints))]
        self.column_domains = [[] for _ in range(len(self.column_hints))]
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
        for i, c in enumerate(self.row_hints):
            for _ in c:
                self.row_domains[i].append([x for x in range(self.shape[0])])
        for i, c in enumerate(self.column_hints):
            for _ in c:
                self.column_domains[i].append([x for x in range(self.shape[1])])

    def reduce_domains(self):
        for i, (domains, hints) in enumerate(zip(self.row_domains, self.row_hints)):
            for j, d in enumerate(domains):
                pass


    @staticmethod
    def revise(variable, var_lengths, var_min, n_vars):
        if variable < n_vars:
            ModelV3.revise(variable+1, var_lengths, var_lengths[var_lengths]+var_min, n_vars)


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
