


class Model:

    def __init__(self, path):
        specs = self.read_file(path)
        self.shape = specs[0]
        self.row_constraints = list(reversed(specs[1:self.shape[1]]))
        self.column_constraints = list(reversed(specs[1+self.shape[1]:]))
        self.domains = [[[]*self.shape[0]],[[]*self.shape[1]]]

    def h(self):
        pass

    def generate_next(self):
        pass

    def generate_domains(self):
        for variables in self.column_constraints:
            while variables[-1] < self.shape[0]:
                pass

    @staticmethod
    def read_file(path):
        with open(path, 'r') as f:
            ret = []
            for l in f:
                ret.append([int(x) for x in l.split(' ')])
        return ret
