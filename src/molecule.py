class Molecule:
    def __init__(self, graph, nodes):
        self.graph = graph.to('cuda') # Ajacency Matrix
        self.nodes = nodes.to('cuda') # means C, N, ... Br
        
