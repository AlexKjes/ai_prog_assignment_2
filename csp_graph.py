

class Node:
    def __init__(self, domain):
        self.domain = domain
        self.to_edges = []

    def update(self, new_domain):
        self.domain = new_domain
        for e in self.to_edges:
            e.revise()


class Edge:
    def __init__(self, from_node, to_node, constraint):
        self.to_node = to_node
        to_node.to_edges.append(self)
        self.from_node = from_node
        self.constraint = constraint
        self.revise()

    def revise(self):
        updated_domain = []
        change = False
        for element in self.from_node.domain:
            if len(self.to_node.domain) != 0 and self.constraint(element, self.to_node.domain):
                updated_domain.append(element)
            else:
                change = True
        if change:
            self.from_node.update(updated_domain)



if __name__ == '__main__':
    n1 = Node([i for i in range(10)])
    n2 = Node([i for i in range(10)])
    n3 = Node([i for i in range(10)])

    Edge(n1, n2, lambda x, y: any([x < z for z in y]))
    Edge(n2, n3, lambda x, y: x < max(y))
    Edge(n3, n2, lambda x, y: x > min(y))
    Edge(n2, n1, lambda x, y: x > min(y))
    Edge(n1, n1, lambda x, y: x > 3)

    print(n1.domain)
