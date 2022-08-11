from model import CNN
import torch
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from torch.nn import ModuleList
import sys
import torchvision
from torchvision import datasets, transforms
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
trans=transforms.Compose([transforms.Resize(28,28),
                          transforms.ToTensor()])
test_data=torchvision.datasets.ImageFolder(root='./test/',transform=trans)
batch_size=10
model=CNN()
def test(data_loader,model):
    model.eval()
    n_predict=0
    n_correct=0
    with torch.no_grad():
        for X,Y in tqdm.tqdm(data_loader,desc='data_loader'):
            y_hat=model(X)
            y_hat.argmax()
            
            _,predicted=torch.max(y_hat,1)
            
            n_predict += len(predicted)
            n_correct += (Y==predicted).sum()
            
    accuracy=int(n_correct)/n_predict 
    print(f"Accuracy:{accuracy}()")

model.load_state_dict(torch.load('./model.pt'))
model.eval()
resultLoader=torch.utils.data.DataLoader(test_data,shuffle=False)

test(resultLoader,model)
sys.stdout = open('result.txt','w')
for data in resultLoader:
    inputs,labels=data
    outputs=model(inputs)
    _,predicted=torch.max(outputs,1)
    print(predicted.item())

