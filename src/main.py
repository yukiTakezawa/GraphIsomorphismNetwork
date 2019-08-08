import sys
import numpy as np
import math
import random
from functions import *
import matplotlib.pyplot as plt


#*_graph.txtを開きリストに変換
def file2matrix(file_path, label_path):
    file = open(file_path)
    
    lines = file.readlines()
    file.close()
    
    data = []
    for line in lines:
        data.append([int(n) for n in line.split()])
        

    size = data.pop(0)
    matrix = np.array(data)

    #labelを取得
    label_file = open(label_path)
    lines = label_file.readlines()
    label_file.close()
    data = []
    for line in lines:
        data.append([int(n) for n in line.split()])
    
    return data[0][0], matrix

class GraphNeuralNet:
    def __init__(self, node_dim):
        self.node_dim = node_dim
        self.param = {}
        self.param['W'] = np.random.normal(loc=0, scale=1.0, size=(node_dim, node_dim))
        self.param['A'] = np.random.normal(loc=0, scale=1.0, size=node_dim)
        self.param['b'] = 0 #np.random.normal()

    def forward(self, graph, nodes):
        a = np.zeros((len(nodes), self.node_dim)) 
        for i in range(len(nodes)):
            for j in range(int(len(nodes))):
                for d in range(self.node_dim):
                    a[i][d] += nodes[i][d] * graph[i][j]
        next_nodes = np.zeros((len(nodes), self.node_dim))
        for i in range(len(nodes)):
            for d in range(self.node_dim):
                tmp = 0
                for k in range(self.node_dim):
                    tmp += self.param['W'][d][k] * a[i][k]
                next_nodes[i][d] = np.maximum(0, tmp) #ReLU
        #\print(graph)
        #print(next_nodes)        
        return next_nodes
    
        
    def predict(self, graph, nodes):
        next_nodes = self.forward(graph, nodes)
        next_nodes = self.forward(graph, next_nodes)
        h = np.zeros(self.node_dim)
        for i in range(next_nodes.shape[0]):
            for d in range(self.node_dim):
                h[d] = next_nodes[i][d]
                
        sum = 0
        for i in range(self.node_dim):
            sum += (self.param['A'])[i] * h[i]
        sum += self.param['b']
        return sigmoid(sum)

    def loss(self, graph, nodes, label):
        p = self.predict(graph, nodes)
        #return ((label - p)*5)**2
        
        return -label*math.log(p) - (1.0 - label)*math.log(1.0 - p)
        #return label*math.log(1.0 + math.exp(-p)) + (1.0 - label)*math.log(1.0 + math.exp(p))
        
    def gradient(self, graph, nodes, label):
        h = 0.001
        grad = {}
        grad['W'] = np.zeros_like(self.param['W'])
        grad['A'] = np.zeros_like(self.param['A'])
        grad['b'] = np.zeros_like(self.param['b'])
        
        for i in range(self.param['W'].shape[0]):
            for j in range(self.param['W'].shape[1]):
                tmp = self.param['W'][i][j]
                self.param['W'][i][j] = tmp + h
                d_loss1 = self.loss(graph, nodes, label)
                self.param['W'][i][j] = tmp - h
                d_loss2 = self.loss(graph, nodes, label)
                self.param['W'][i][j] = tmp
                grad['W'][i][j] = (d_loss1 - d_loss2) / (2*h)
        for i in range(self.param['A'].size):
            tmp = self.param['A'][i]
            self.param['A'][i] = tmp + h
            d_loss1 = self.loss(graph, nodes, label)
            self.param['A'][i] = tmp - h
            d_loss2 = self.loss(graph, nodes, label)
            self.param['A'][i] = tmp
            grad['A'][i] = (d_loss1 - d_loss2) / (2*h)
        #bに関して微分
        tmp = self.param['b']
        self.param['b'] = tmp + h
        d_loss1 = self.loss(graph, nodes, label)
        self.param['b'] = tmp - h
        d_loss2 = self.loss(graph, nodes, label)
        self.param['b'] = tmp
        grad['b'] = (d_loss1 - d_loss2) / (2.0*h)
        
        return grad
        
def calc_accuracy(graph_nn):
       
    test_index_list = list(range(100))
    accuracy = 0.0
    print("calcurate accuracy")
    for index in test_index_list:
        
        file_path = './datasets/train/' + str(index) + '_graph.txt'
        label_path = './datasets/train/' + str(index) + '_label.txt'
            
        label, graph = file2matrix(file_path, label_path)
        graph_size = graph.shape
        
        nodes = np.zeros((graph_size[0], D)) #nodeの重さをランダムに生成
        for i in range(graph_size[0]):
            for j in range(D):
                nodes[i][j] = 1.0 #np.random.normal(loc=10)

        p = graph_nn.predict(graph, nodes)
        if (p>=0.5) & (label == 1):
            accuracy += 1.0
        if (p<0.5) & (label == 0):
            accuracy += 1.0
    return accuracy/100.0
        
if __name__ == '__main__':
    plt.ion()
    
    D = 5 #nodeの重みの次元
    batch_size = 25
    epoch_size = 100
    graph_nn = GraphNeuralNet(D)
    learning_rate = 0.1
    index_list = list(range(100))
    loss_list = []
    current_epoch_num = 0

    # Momentum-SGDのために準備
    # 前のパラメータの更新料を保存
    momentum = 0.7
    prev_param_update_val = {}
    prev_param_update_val['W'] = np.zeros(D)
    prev_param_update_val['A'] = np.zeros(D)
    prev_param_update_val['b'] = 0


    print(calc_accuracy(graph_nn))
    
    for trial in range(epoch_size):
        print('train next epoch %d' % current_epoch_num)
        current_epoch_num += 1
        mini_batch_index_list = random.sample(index_list, len(index_list))

        while (len(mini_batch_index_list) > 0):
            #print("train next mini batch")
            sum_of_grad = {}
            sum_of_grad['W'] = np.zeros((D, D))
            sum_of_grad['A'] = np.zeros(D)
            sum_of_grad['b'] = 0
            
            for i in range(batch_size):
                index = mini_batch_index_list.pop()
                file_path = './datasets/train/' + str(index) + '_graph.txt'
                label_path = './datasets/train/' + str(index) + '_label.txt'
                
                label, graph = file2matrix(file_path, label_path)
                graph_size = graph.shape
            
                nodes = np.zeros((graph_size[0], D)) #nodeの重さをランダムに生成
                for i in range(graph_size[0]):
                    for j in range(D):
                        nodes[i][j] = 1.0 #np.random.normal(loc=10)

                loss_list.append(graph_nn.loss(graph, nodes, label))
                #print(graph_nn.loss(graph, nodes, label)) 
                grad = graph_nn.gradient(graph, nodes, label)
                sum_of_grad['W'] += grad['W']
                sum_of_grad['A'] += grad['A']
                sum_of_grad['b'] += grad['b']
                
            graph_nn.param['W'] -= learning_rate*sum_of_grad['W'] / batch_size + momentum*prev_param_update_val['W']
            graph_nn.param['A'] -= learning_rate*sum_of_grad['A'] / batch_size + momentum*prev_param_update_val['A']
            graph_nn.param['b'] -= learning_rate*sum_of_grad['b'] / batch_size + momentum*prev_param_update_val['b']
            
            #動的にグラフを描画
            plt.plot(loss_list)
            plt.draw()
            plt.pause(0.00001)

        
        print(calc_accuracy(graph_nn))
        
    #plt.show()
    plt.savefig('loss_transition.png')
    print(graph_nn.param)

    print(calc_accuracy(graph_nn))
