import numpy as np
import tkinter as tk
from csp_graph import *



class Model:

    def __init__(self, path):
        # Read file n stuff
        specs = self.read_file(path)
        self.csp_graph = Graph()
        self.shape = specs[0]
        self.row_hints = specs[self.shape[1]:0:-1]
        self.column_hints = specs[self.shape[1]+1:]

        # initialize hint variables
        self.row_variables = [[] for _ in range(len(self.row_hints))]
        self.column_variables = [[] for _ in range(len(self.column_hints))]
        self.generate_domains_and_1d_constraints()

        # Generate row sequences
        self.row_sequences = [[] for _ in self.row_hints]
        self.generate_row_sequences()
        self.column_sequences = [[] for _ in self.column_hints]
        self.generate_column_sequences()

        # re init csp
        self.csp_graph = Graph()
        self.row_variables = []
        self.column_variables = []
        self.sequences_to_csp_vars()
        self.column_constraints = []
        self.row_constraints = []

        self.generate_2d_row_seq_constraints()
        self.generate_2d_column_seq_constraints()

        self.csp_graph.save_state_by_key(self.generate_state_id())

        self.tk_master = None
        self.tk_canvas = None

    def h(self, state):
        self.csp_graph.load_state_by_key(state)
        ret = 0
        for n in self.csp_graph.nodes:
            ret += len(n.domain)
        return ret

    def generate_next(self, state):
        ret = []
        for n in self.row_variables:
            for i, d in enumerate(n.domain):
                self.csp_graph.load_state_by_key(state)
                n.update([d])
                if not self.is_invalid():
                    state_id = self.generate_state_id()
                    ret.append(state_id)
                    self.csp_graph.save_state_by_key(state_id)
        return ret

    def generate_state_id(self):
        ret = []
        for n in self.csp_graph.nodes:
            ret.append(tuple([tuple(d) for d in n.domain]))
        return tuple(ret)

    def is_solution(self, state):
        self.csp_graph.load_state_by_key(state)
        for vs in self.row_variables:
            if len(vs.domain) != 1:
                return False

        img = self.fill_matrix()
        for y, r in enumerate(img):
            segment = 0
            segment_length = 0
            in_segment = False
            for x, p in enumerate(r):
                if in_segment:
                    if p == 1:
                        segment_length += 1
                    else:
                        if segment_length != self.row_hints[y][segment]:
                            return False
                        segment += 1
                        in_segment = False
                else:
                    if p == 1:
                        in_segment = True
                        segment_length = 1
        for x, c in enumerate(img.T):
            segment = 0
            segment_length = 0
            in_segment = False
            for y, p in enumerate(c):
                if segment == len(self.column_hints[x]):
                    break
                if in_segment:
                    if p == 1:
                        segment_length += 1
                    else:
                        if segment_length != self.column_hints[x][segment]:
                            return False
                        segment += 1
                        in_segment = False
                else:
                    if p == 1:
                        in_segment = True
                        segment_length = 1

        return True

    def is_invalid(self):
        for v in self.csp_graph.nodes:
            if len(v.domain) == 0:
                return True
        return False

    def generate_row_sequences(self):
        for y, row in enumerate(self.row_variables):
            [rv.save() for rv in row]
            self.generate_row_sequence(y, 0)
            [rv.revert() for rv in row]

    def generate_row_sequence(self, y, i):
        row = self.row_variables[y]
        for dc in row[i].domain:
            [r.save() for r in row]
            row[i].update([dc])
            if i < len(row)-1:
                self.generate_row_sequence(y, i+1)
            else:
                row_seq = np.zeros(len(self.column_hints), dtype=int)
                for j, rv in enumerate(row):
                    d = rv.domain[0]
                    row_seq[d:d+self.row_hints[y][j]] = 1
                self.row_sequences[y].append(row_seq)
            [r.revert() for r in row]

    def generate_column_sequences(self):
        for x, column in enumerate(self.column_variables):
            [cv.save() for cv in column]
            self.generate_column_sequence(x, 0)
            [cv.revert() for cv in column]

    def generate_column_sequence(self, x, i):
        column = self.column_variables[x]
        for dc in column[i].domain:
            [c.save() for c in column]
            column[i].update([dc])
            if i < len(column)-1:
                self.generate_column_sequence(x, i+1)
            else:
                column_seq = np.zeros(len(self.row_hints), dtype=int)
                for j, cv in enumerate(column):
                    d = cv.domain[0]
                    column_seq[d:d+self.column_hints[x][j]] = 1
                self.column_sequences[x].append(column_seq)
            [c.revert() for c in column]

    def sequences_to_csp_vars(self):
        for row in self.row_sequences:
            node = self.csp_graph.add_node(row)
            self.row_variables.append(node)
        for column in self.column_sequences:
            node = self.csp_graph.add_node(column)
            self.column_variables.append(node)

    def generate_2d_row_seq_constraints(self):
        for i, r in enumerate(self.row_variables):
            def f(row, columns, y=i):
                for x, p in enumerate(row):
                    l = []
                    for c in columns[x]:
                        if c[y] not in l:
                            l.append(c[y])
                    if p not in l:
                        return False
                return True
            self.row_constraints.append(self.csp_graph.add_edge(r, self.column_variables, f))

    def generate_2d_column_seq_constraints(self):
        for j, c in enumerate(self.column_variables):
            def g(column, rows, x=j):
                for y, p in enumerate(column):
                    l = []
                    for r in rows[y]:
                        if r[x] not in l:
                            l.append(r[x])
                        if len(l) == 2:
                            break
                    if p not in l:
                        return False
                return True
            self.column_constraints.append(self.csp_graph.add_edge(c, self.row_variables, g))

    def generate_2d_row_seq_constraints_fuzz(self):
        for i, r in enumerate(self.row_variables):
            def f(row, columns, y=i):
                truth = [False] * self.shape[0]
                for x, p in enumerate(row):
                    l = []
                    for c in columns[x]:
                        if c[y] not in l:
                            l.append(c[y])
                    if p in l:
                        truth[x] = True
                if sum(truth)/len(truth) > .8:
                    return True
                else:
                    return False
            self.row_constraints.append(self.csp_graph.add_edge(r, self.column_variables, f))

    def generate_2d_column_seq_constraints_fuzz(self, threshold):
        for j, c in enumerate(self.column_variables):
            def g(column, rows, x=j):
                truth = [False] * self.shape[1]
                for y, p in enumerate(column):
                    l = []
                    for r in rows[y]:
                        if r[x] not in l:
                            l.append(r[x])
                        if len(l) == 2:
                            break
                    if p in l:
                        truth[y] = True
                if sum(truth)/len(truth) > threshold:
                    return True
                else:
                    return False
            self.column_constraints.append(self.csp_graph.add_edge(c, self.row_variables, g))

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
                if j == len(column)-1:  # if last variable
                    self.csp_graph.add_edge(self.column_variables[i][j], [self.column_variables[i][j]], lambda x, y, l=h: x+l-1 < self.shape[1])
                else:
                    self.csp_graph.add_edge(self.column_variables[i][j], [self.column_variables[i][j+1]], lambda x, y, l=h: x+l < max(y[0]))
                if j != 0:  # if not first variable
                    self.csp_graph.add_edge(self.column_variables[i][j], [self.column_variables[i][j-1]], lambda x, y, lp=column[j-1]: x > min(y[0])+lp)

    def fill_matrix(self):
        ret = np.zeros((self.shape[1], self.shape[0]))
        for i, row in enumerate(self.row_variables):
                ret[i] = row.domain[0]
        return ret

    def draw_state(self, state):
        scale = 55 - self.shape[1]
        self.csp_graph.load_state_by_key(state)
        if self.tk_master is None:
            self.tk_master = tk.Tk()
            self.tk_canvas = tk.Canvas(self.tk_master, width=self.shape[0] * scale, height=self.shape[1] * scale)
            self.tk_canvas.pack()

        self.tk_canvas.delete('all')
        self.tk_canvas.create_rectangle(0, 0, self.shape[0] * scale, self.shape[1] * scale, fill='white')

        state_matrix = self.fill_matrix()
        for y, r in enumerate(state_matrix):
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


