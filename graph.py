class Graph:

    def __init__(self, start_node, node_expander):
        """
        A graph
        :param start_node: State of a start node
        :param node_expander: A function to generate a nodes proceeding nodes
        """
        self.node_expander = node_expander
        self.nodes = {start_node: self.Node(self, start_node)}
        self.n_nodes_generated = 0

    def get_start_node(self):
        """
        :return: A random node
        """
        return next(iter(self.nodes.values()))

    class Node:
        def __init__(self, graph, state):
            """
            :param Graph graph: the graph in which this node lives
            :param hashable state: the unique id of the node

            :var [Node] self.parents: All preceding nodes
            :var [Node] self._children: All proceeding nodes
            :var boolean self.is_expanded: A flag to tell if the node has had its children generated
            """
            self._graph = graph

            self.state = state

            self.parents = []
            self._children = []
            self.is_expanded = False

        def get_children(self):
            """
            Generates children, if it hasn't already been done and returns them.
            :return [Node]: list of children
            """
            if not self.is_expanded:
                self.is_expanded = True
                self._expand()
            return self._children

        def _expand(self):
            """
            1. Expands this node.
            2. Updates the child's parents if it is known
            3. Initializes the child if it is new to the graph
            :return: None
            """
            children = self._graph.node_expander(self.state)  # 1
            for child in children:
                self._graph.n_nodes_generated += 1
                if child in self._graph.nodes:  # 2
                    self._graph.nodes[child].parents.append(self)
                    self._children.append(self._graph.nodes[child])
                else:  # 3
                    cn = Graph.Node(self._graph, child)
                    cn.parents.append(self)
                    self._graph.nodes[child] = cn
                    self._children.append(cn)



