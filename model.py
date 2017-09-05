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

        self.tk_master = None
        self.tk_canvas = None

    def h(self):
        pass

    def generate_next(self):
        pass

    def is_solution(self):
        for vs in self.row_variables + self.column_variables:
            for v in vs:
                if len(v.domain) != 1:
                    return False
        return True

    def is_invalid(self):
        for vs in self.row_variables + self.column_variables:
            for v in vs:
                if len(v.domain) == 0:
                    return True
        return False


    def generate_domains_and_1d_constraints(self):
        for i, row in enumerate(self.row_hints):
            for _ in row:
                self.row_variables[i].append(self.csp_graph.add_node([x for x in range(self.shape[0])]))
            for j, h in enumerate(row):
                if j == len(row)-1:  # if last variable
                    self.csp_graph.add_edge(self.row_variables[i][j], [self.row_variables[i][j]], lambda x, y, l=h: x+l-1 < self.shape[0])
                else:
                    self.csp_graph.add_edge(self.row_variables[i][j], [self.row_variables[i][j+1]], lambda x, y, l=h: x+l < max(y[0]))
                if j != 0:  # if not first variable
                    self.csp_graph.add_edge(self.row_variables[i][j], [self.row_variables[i][j-1]], lambda x, y, lp=row[j-1]: x > min(y[0])+lp)
        for i, column in enumerate(self.column_hints):
            for _ in column:
                self.column_variables[i].append(self.csp_graph.add_node([x for x in range(self.shape[1])]))
            for j, h in enumerate(column):
                #if i == 4 and j == 0:
                #    print(h)
                #    print(self.column_variables[i][j].domain)
                #    print(self.column_variables[i][j+1].domain)
                if j == len(column)-1:  # if last variable
                    self.csp_graph.add_edge(self.column_variables[i][j], [self.column_variables[i][j]], lambda x, y, l=h: x+l-1 < self.shape[1])
                else:
                    self.csp_graph.add_edge(self.column_variables[i][j], [self.column_variables[i][j+1]], lambda x, y, l=h: x+l < max(y[0]))
                if j != 0:  # if not first variable
                    self.csp_graph.add_edge(self.column_variables[i][j], [self.column_variables[i][j-1]], lambda x, y, lp=column[j-1]: x > min(y[0])+lp)

    def generate_2d_constraints(self):

        for i, cvs in enumerate(self.column_variables):
            for col_var, h in zip(cvs, self.row_hints[i]):
                def f(v, w, h=h, x=i):
                    row_truth = [False] * h
                    for d, (rvs, lengths) in enumerate(zip(self.row_variables[v:v+h], self.row_hints[v:v+h])):
                        for rv, l in zip(rvs, lengths):
                            for r in rv.domain:
                                if x in [seq for seq in range(r, r+l)]:
                                    row_truth[d] = True
                                    break
                    return all(row_truth)
                rvs = []
                [[rvs.append(ugh) for ugh in barf] for barf in self.row_variables]
                self.csp_graph.add_edge(col_var, rvs, f)

        for i, rvs in enumerate(self.row_variables):
            for row_var, h in zip(rvs, self.row_hints[i]):
                def ff(v, w, h=h, y=i):
                    column_truth = [False] * h
                    for d, (cvs, lengths) in enumerate(zip(self.column_variables[v:v+h], self.column_hints[v:v+h])):
                        for cv, l in zip(cvs, lengths):
                            for c in cv.domain:
                                if y in [seq for seq in range(c, c+l)]:
                                    column_truth[d] = True
                                    break
                    return all(column_truth)
                cvs = []
                [[cvs.append(ugh) for ugh in barf] for barf in self.column_variables]
                self.csp_graph.add_edge(row_var, cvs, ff)

                #self.csp_graph.add_edge(rv, cvs, lambda v, w, d=x: v != d or [any([any([v == l for l in range(dc, dc+self.column_hints[d][i])]) for dc in t] for i, t in enumerate(w))])




    def generate_segments(self, constraints, domain):
        pass

    def fill_matrix(self):
        ret = np.zeros(self.shape)
        for y, (rvs, hs) in enumerate(zip(self.row_variables, self.row_hints)):
            for r, h in zip(rvs, hs):
                for x in range(r.domain[0], r.domain[0]+h):
                    ret[x][y] = 1

        return ret.T


    def draw_state(self, state):
        scale = 50
        if self.tk_master is None:
            self.tk_master = tk.Tk()
            self.tk_canvas = tk.Canvas(self.tk_master, width=self.shape[0] * scale, height=self.shape[1] * scale)
            self.tk_canvas.pack()

        self.tk_canvas.delete('all')
        self.tk_canvas.create_rectangle(0, 0, self.shape[0] * scale, self.shape[1] * scale, fill='white')

        for y, r in enumerate(state):
            for x, p in enumerate(r):
                if p == 1:
                    self.tk_canvas.create_rectangle(x*scale, y*scale, (x+1)*scale, (y+1)*scale, fill='red')

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
