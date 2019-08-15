import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt

from functions import *
from gnn import *

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gin = GraphIsomorphismNetwork(4, 20) 
        self.fun1 = nn.Linear(4, 50)
        self.fun2 = nn.Linear(50, 25)
        self.fun3 = nn.Linear(25, 1)

    def forward(self, molecule):
        x = self.gin.predict(molecule)
        #x = x / x.max()
        x = self.fun1(x)
        x = F.sigmoid(x)
        x = self.fun2(x)
        x = F.sigmoid(x)
        x = self.fun3(x)
        x = F.sigmoid(x)
        #print(x)
        return x

def preproceccing(data):
    maximum_node_val = 0 # maximum value of node value
    maximum_attribute_val = 0 # maximum value of attribute value 
    for i in range(len(data)):
        tmp = data[i].nodes[:,0:3].max()
        tmp = tmp.cpu()
        tmp = tmp.numpy()
        if (tmp > maximum_node_val):
            maximum_node_val = tmp

        tmp = data[i].nodes[:,3:4].max()
        tmp = tmp.cpu()
        tmp = tmp.numpy()
        if (tmp > maximum_attribute_val):
            maximum_attribute_val = tmp
            
    for i in range(len(data)):
        data[i].nodes[:,0:3] /= torch.from_numpy(maximum_node_val)
        data[i].nodes[:,3:4] /= torch.from_numpy(maximum_attribute_val)
    
        
    return data
            
def shuffle_data(x, y):
    zipped = list(zip(x, y))
    np.random.shuffle(zipped)
    x, y = zip(*zipped)
    return x, y

def calc_accuracy(model, test_data, test_labels, test_size):
    correct = 0
    error1 = 0
    error0 = 0
    count1 = 0
    count0 = 0
    with torch.no_grad():
        for i in range(test_size):
            output = model(test_data[i])
            prediction = (output.cpu()).numpy()[0]
            label = (test_labels[i].cpu()).numpy()[0]
            
            if ((prediction>=0.5) & (label==1)):
                correct += 1
                count1 += 1
            elif ((prediction<0.5) & (label==0)):
                correct += 1
                count0 += 1
            elif ((prediction>=0.5) & (label==0)):
                error1 += 1
            else:
                error0 += 1
                
    print(count0, count1, error0, error1)
    return correct/test_size
    
def main():
    epoch_size = 300
    data_size = 1112
    accuracy_list = []
    
    device = torch.device('cuda')
    data, labels = load_data()
    model = Model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.BCELoss()#nn.MSELoss()

    torch.autograd.set_detect_anomaly(True)
    
    for i in range(epoch_size):

        shuffled_data, shuffled_labels = shuffle_data(data, labels)

        
        for j in range(data_size):
            with torch.autograd.detect_anomaly():
                optimizer.zero_grad()
                output = model(shuffled_data[j])
                #print(output, shuffled_labels[j])
                loss = criterion(output, shuffled_labels[j])
                loss.backward()
                optimizer.step()
                
        accuracy = calc_accuracy(model, data, labels, data_size)
        accuracy_list.append(accuracy)
        print(accuracy)
        print("learnig %d epoch" % (i))
    print("finished learning \n calcurating accuracy")


    plt.plot(accuracy_list)
    plt.show()
    print("end")

if __name__ == '__main__':
    main()
