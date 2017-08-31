

class BestFirst:

    def __init__(self, graph, h_fn, solution_fn):
        """
        :param graph: The graph containing the search space
        :param h_fn: A heuristic function that takes a node state and returns a scalar
        :param solution_fn: A function that determines if a solution state has been found

        :var int self.expand_counter: Counts how many nodes that has been expanded
        :var dict self.frontier: Keeps track of unexpanded nodes
        :var node self.last_expanded: The last node to be expanded
        :var [node] self.solution: A list of the solution path
        """

        self.h_fn = h_fn
        self.solution_fn = solution_fn

        self.expand_counter = 0

        self.graph = graph
        # initialize first node
        node = graph.get_start_node()
        node.g = 0
        node.h = h_fn(node.state)
        node.prev = None

        self.frontier = {node.state: node}
        self.last_expanded = None
        self.solution = None

    def next(self):
        """
        1. Draws the current best node to expand.
        2. Assigns the node to self.last_expanded. Not important, but needed to satisfy assignment criteria.
        3. Checks if its a solution.
        4. Increment expand counter.
        5. Iterate over the expanded nodes children.
        6. Check if the node has been visited before.
        7. Update previously visited node.
        8. Initialize unvisited node.
        :return: None
        """
        next_node = self.get_next_best_node()  # 1
        self.last_expanded = next_node  # 2
        if self.solution_fn(next_node.state):  # 3
            print("solution_ found")
            self.solution = self.generate_solution_path(next_node)
            return
        self.expand_counter += 1  # 4
        for cn in next_node.get_children():  # 5
            if hasattr(cn, 'g'):  # 6
                self.update_g(cn, next_node)  # 7
            else:
                cn.prev = next_node           # |
                cn.g = next_node.g+1          # |
                cn.h = self.h_fn(cn.state)    # | 8
                self.frontier[cn.state] = cn  # |

    def get_next_best_node(self):
        """
        Finds the best candidate to expand
        :return: Node
        """
        best_node = next(iter(self.frontier.values()))
        best_w = best_node.g + best_node.h
        for n in self.frontier.values():
            nw = n.g + n.h
            if nw < best_w:
                best_w = nw
                best_node = n
        del self.frontier[best_node.state]
        return best_node

    def update_g(self, node, preceding_node):
        """
        Updates a previously visited nodes g value if it is less than previously set,
        and updates its unexpanded children.
        :param node: A previously visited node
        :param preceding_node: The nodes preceding node
        :return: None
        """
        if node.g > preceding_node.g + 1:
            node.g = preceding_node.g + 1
            node.prev = preceding_node
            if node.is_expanded:
                for cn in node.get_children():
                    self.update_g(cn, node)

    @staticmethod
    def generate_solution_path(solution_node):
        """
        :param solution_node: The node that satisfied termination criteria
        :return [node]: returns nodes in the path from origin node to solution
        """
        """
        ret = []
        current = solution_node
        while True:
            ret.insert(0, current)
            current = current.prev
            if current.prev is None:
                break
        return ret
        """
        ret = []
        current = solution_node
        while True:
            ret.insert(0, current)
            parents = current.parents
            sorted(parents, key=lambda node: node.g)
            current = parents[0]
            if current.g == 0:
                break
        return ret






