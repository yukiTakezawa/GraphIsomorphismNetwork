import math
import torch

def load_data():
    data_num = 188
    
    edge_file = open('../dataset/Mutag/Mutag.edges')
    graph_labels_file = open('../dataset/Mutag/Mutag.graph_labels')
    node_labels_file = open('../dataset/Mutag/Mutag.node_labels')
    graph_idx_file = open('../dataset/Mutag/Mutag.graph_idx')
    link_labels_file = open('../dataset/Mutag/Mutag.link_labels')

    # check node size
    tmp0, tmp1 = 1, 1
    for graph_idx in range(data_num):
        node_size = 0

        while True:
            tmp1 = graph_idx_file.readlines()
            if (tmp0 == tmp1):
                node_size += 1
                tmp0 = tmp1
            else:
                tmp0 = tmp1
                break

        # check ajacency matric
        graph = torch.zeros(node_size, node_size)
        
    
def sigmoid(s):
    return 1.0/(1.0 + math.exp(-s))

