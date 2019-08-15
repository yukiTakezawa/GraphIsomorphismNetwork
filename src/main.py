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
        self.fun0 = nn.Linear(4, 100)
        self.fun1 = nn.Linear(100, 50)
        self.fun2 = nn.Linear(50, 25)
        self.fun3 = nn.Linear(25, 1)

    def forward(self, x):
        #print(x)
        x = self.fun0(x)
        x = F.relu(x)
        x = self.fun1(x)
        x = F.relu(x)
        x = self.fun2(x)
        x = F.relu(x)
        x = self.fun3(x)
        x = F.sigmoid(x)
        #print(x)
        return x


def preproceccing(data):
    ### processe data by GIN and normalize each value
    # 
    # Args:
    #   data(list of Molecule):
    #
    # Returns:
    #   list of torch.Tensor : processed data by GIN
    ###

    gin = GraphIsomorphismNetwork(4, 20)
    processed_data = []
    
    for i in range(len(data)):
        processed_data.append(gin.predict(data[i]))
        
    maximum_attribute_val = 0 # maximum value of attribute value 
        
    for i in range(len(processed_data)):
        processed_data[i][0:3] /= processed_data[i][0:3].max()
        
        tmp = processed_data[i][3]
        tmp = tmp.cpu()
        tmp = tmp.numpy()
        if (tmp > maximum_attribute_val):
            maximum_attribute_val = tmp

    for i in range(len(processed_data)):
        processed_data[i][3] /= torch.from_numpy(maximum_attribute_val)

    return processed_data
            
def shuffle_data(x, y):
    ### shuffle x and y
    #
    # Args:
    #   x (list):
    #   y (list);
    #
    # Returns:
    #   list, list: shuffled x and y
    ###
    
    zipped = list(zip(x, y))
    np.random.shuffle(zipped)
    x, y = zip(*zipped)
    return x, y

def calc_accuracy(model, test_data, test_labels, test_size):
    ### calcurate accuracy
    #
    # Args:
    #    model (Model): trained model
    #    test_data (list): 
    #    test_labels (list):
    #    test_size (int): size of test data
    #
    # Returns:
    #   float: accuracy
    ###
    
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
    epoch_size = 1500
    train_data_size = 912
    test_data_size = 200
    accuracy_list = []
    
    device = torch.device('cuda')
    data, labels = load_data()
    processed_data = preproceccing(data)
    
    model = Model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.BCELoss()#nn.MSELoss()

    torch.autograd.set_detect_anomaly(True)

    # split data into train data and test data
    shuffled_data, shuffled_labels = shuffle_data(processed_data, labels)
    test_data = shuffled_data[0:200]
    test_labels = shuffled_labels[0:200]
    train_data = shuffled_data[200:1112]
    train_labels = shuffled_labels[200:1112]

    # traning
    for i in range(epoch_size):

        shuffled_data, shuffled_labels = shuffle_data(train_data, train_labels)
        
        for j in range(train_data_size):
            with torch.autograd.detect_anomaly():
                optimizer.zero_grad()
                output = model(shuffled_data[j])
                #print(output, shuffled_labels[j])
                loss = criterion(output, shuffled_labels[j])
                loss.backward()
                optimizer.step()
                
        accuracy = calc_accuracy(model, test_data, test_labels, test_data_size)
        accuracy_list.append(accuracy)
        print(accuracy)
        print("learnig %d epoch" % (i))
    print("finished learning \n best accuracy is %f" % (max(accuracy_list)))


    plt.plot(accuracy_list)
    plt.show()
    print("end")

if __name__ == '__main__':
    main()
