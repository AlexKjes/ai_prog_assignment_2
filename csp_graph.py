from copy import  copy


class Graph:
    """
    Constraint satisfaction graph contains variables connected by constraints
    """
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, domain):
        n = Node(domain)
        self.nodes.append(n)
        return n

    def add_edge(self, from_node, to_node, constraint):
        e = Edge(from_node, to_node, constraint)
        self.edges.append(e)
        return e

    def save_state(self):
        """
        saves the current state of the graph
        :return:
        """
        for n in self.nodes:
            n.save()


    def save_state_by_key(self, key):
        """
        saves the current state of the graph by  key, so it can be retrieved by the same key
        :param key:
        :return:
        """
        for n in self.nodes:
            n.key_save(key)

    def revert(self):
        for n in self.nodes:
            n.revert()

    def load_state_by_key(self, key):
        for n in self.nodes:
            n.key_revert(key)


class Node:
    """
    Node represents a variable in a csp
    """
    def __init__(self, domain):
        self.domain = domain
        self.history = []
        self.name_store = {}
        self.to_edges = []

    def update(self, new_domain):
        """
        updates to this variables domain notifies reevaluates variables dependant on this
        :param new_domain:
        :return:
        """
        self.domain = new_domain
        for e in self.to_edges:
            e.revise()

    def save(self):
        self.history.append(copy(self.domain))

    def key_save(self, key):
        self.name_store[key] = copy(self.domain)

    def revert(self):
        self.domain = self.history.pop()

    def key_revert(self, key):
        self.domain = self.name_store[key]


class Edge:
    """
    Represents a constraint
    """
    def __init__(self, from_node, to_nodes, constraint):
        self.to_nodes = to_nodes
        [to_node.to_edges.append(self) for to_node in to_nodes]
        self.from_node = from_node
        self.constraint = constraint
        self.revise()

    def revise(self):
        updated_domain = []
        change = False
        for element in self.from_node.domain:
            if all([len(to_node.domain) != 0 for to_node in self.to_nodes]) and self.constraint(element, [to_node.domain for to_node in self.to_nodes]):
                updated_domain.append(element)
            else:
                change = True
        if change:
            self.from_node.update(updated_domain)

    def remove(self):
        [n.to_edges.remove(self) for n in self.to_nodes]


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




