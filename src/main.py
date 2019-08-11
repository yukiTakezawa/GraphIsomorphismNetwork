import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from functions import *
from gnn import *

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gin = GraphIsomorphismNetwork(4, 3) # if loop_size>=4 then output=nan
        self.fun1 = nn.Linear(4, 20)
        self.fun2 = nn.Linear(20, 30)
        self.fun3 = nn.Linear(30, 1)

    def forward(self, molecule):
        x = self.gin.predict(molecule)
        x = self.fun1(x)
        x = torch.sigmoid(x)
        x = self.fun2(x)
        x = torch.sigmoid(x)
        x = self.fun3(x)
        x = torch.sigmoid(x)
        return x

def shuffle_data(x, y):
    zipped = list(zip(x, y))
    np.random.shuffle(zipped)
    x, y = zip(*zipped)
    return x, y
    
    
def main():
    epoch_size = 10
    data_size = 1112
    
    device = torch.device('cuda')
    data, labels = load_data()
    model = Model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.MSELoss()

    
    for i in range(epoch_size):

        shuffled_data, shuffled_labels = shuffle_data(data, labels)
        
        for j in range(data_size):
            optimizer.zero_grad()
            output = model(shuffled_data[j])
            loss = criterion(output, shuffled_labels[i])
            loss.backward()
            optimizer.step()

        print("learnig %d epoch" % (i))
    print("finished learning \n calcurating accuracy")
    
    correct = 0
    error = 0
    with torch.no_grad():
        for i in range(data_size):
            output = model(data[i])
            prediction = (output.cpu()).numpy()[0]
            label = (labels[i].cpu()).numpy()[0]
            
            if ((prediction>=0.5) & (label==1)):
                correct += 1
            elif ((prediction<0.5) & (label==0)):
                correct += 1
            else:
                print(prediction, label)
                error += 1
                
    print("accuracy is %f" % (correct/data_size))
    print(error)
    
    print("end")

if __name__ == '__main__':
    main()
