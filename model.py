


class Model:

    def __init__(self, path):
        specs = self.read_file(path)
        self.shape = specs[0]
        self.row_constraints = list(reversed(specs[1:self.shape[1]+1]))
        self.column_constraints = list(reversed(specs[1+self.shape[1]:]))
        self.domains = [[[]]*self.shape[0], [[]]*self.shape[1]]
        self.generate_domains()

    def h(self):
        pass

    def generate_next(self):
        pass

    def generate_domains(self):
        for i, constraints in enumerate([self.column_constraints, self.row_constraints]):
            for j, variables in enumerate(constraints):
                k = 0
                while True:
                    l = k
                    d_set = []
                    for v in range(len(variables)):
                        d_set.append(variables[v] + l)
                        l += variables[v] + k + 1
                    k += 1
                    if l < self.shape[i]:
                        self.domains[i][j].append(d_set)
                    else:
                        break


    @staticmethod
    def read_file(path):
        with open(path, 'r') as f:
            ret = []
            for l in f:
                ret.append([int(x) for x in l.split(' ')])
        return ret
