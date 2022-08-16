#!/usr/bin/env python
# coding: utf-8

# Maestría en Explotación de Datos y Gestión del Conocimiento
# 
# Redes Neuronales 
# 
# Prof Leandro Bugnon
# 
# Integrantes:
# 
# *  Di Carlo, M
# *  Ortega, F
# *  Suarez, G
# *  Pastrana, A.
# *  Ortega, V

# In[32]:


import torch as tr 
from torch import nn
import numpy as np 
from matplotlib import pyplot as plt 
import os
import pandas as pd


# In[33]:


from google.colab import drive
drive.mount('/content/drive')


# In[34]:


import sys
sys.path.insert(1,"/content/drive/MyDrive/Austral/Maestria/Redes/Trabajo Final Redes/")
from torch_dataset import MeliDataset


# In[35]:


os.getcwd()
os.chdir(os.path.dirname("/content/drive/MyDrive/Austral/Maestria/Redes/Trabajo Final Redes/"))


# In[36]:


os.getcwd()


# In[37]:


kernel_size = 7
padding = 2
stride = 3
dilation = 5
H = 48
convlayer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, 
                      padding=padding, stride=stride, dilation=dilation)


print(convlayer(tr.ones(1, H, H)).shape)

Hout = (H+2*padding-dilation*(kernel_size-1)-1)/stride + 1

print(int(Hout))


# ## Dataset
# 
# Crearemos un dataset del tipo torch.utils.data.Dataset mediante la clase heredada que se encuentra definida en el torch_dataset.py.

# Transforms es un modulo con funciones que permiten trabajar sobre las imagenes eficientemente. Se deja a continuación las transformaciones mínimas necesarias para entrenar una red: convertir la imagen en un tensor, y normalizar
# 
# Nota: Si las pruebas llevan mucho tiempo, puede ser mejor hacer un conjunto de train más pequeño y usar el conjutno completo solo en los casos más interesantes

# In[38]:


pd.read_csv("to_train.csv").numeric_label.value_counts()


# In[39]:


from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch 

BATCH_SIZE = 32
path_images = "/content/drive/MyDrive/Austral/Maestria/Redes/Trabajo Final Redes/tabla/"       
# TO-DO: Agregar transformaciones para hacer aumentación de datos?
T = transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#####################################  Editar y agregar dataset 
dataset = MeliDataset(csv_file = "to_train.csv",
                      root_dir = "imagenes/MLA1652",
                      transform = transforms.Compose(  
                           [transforms.ToPILImage(),
                            transforms.Resize((60,60)),
                            
                           transforms.ToTensor() ] )
                      )
#####################################
val_size = int(len(dataset)*.2)
test_size = int(len(dataset)*0.1)
train_size = len(dataset) - val_size - test_size
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])


# In[40]:


train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# ## Definición del modelo
# En este caso vamos a crear una clase que hereda de nn.Module. Esto permite encapsular funcionalidades y hacer más simple los ciclos de entrenamiento y evaluación

# In[41]:


import torch.nn as nn
from tqdm.notebook import tqdm
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

class CNNAustral(nn.Module):
    def __init__(self, nclasses, input_channels, device='cpu'):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(18432, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, nclasses)
        )
        
        self.loss_func = nn.CrossEntropyLoss()
        self.optim = tr.optim.Adam(self.parameters(), lr=0.001)

        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.cnn(x)
        x = tr.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
        return x


    def fit(self, loader, verbose=False):
        """Función de entrenamiento (una época)"""
        epoch_loss = 0
        if verbose:
            loader = tqdm(loader)
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)

            yhat = self.forward(x)
            loss = self.loss_func(yhat, y)
            
            epoch_loss += loss.item()

            self.optim.zero_grad()
            loss.backward() 
            self.optim.step()

        return epoch_loss/len(loader)


    def test(self, loader):
        """Función de evaluación (una época)"""
        epoch_loss = 0
        ref, pred = [], []
        for x, y in loader:
            with tr.no_grad():
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.forward(x)

                # predicciones y referencias para calcular tasa de acierto
                ref.append(y.cpu())
                pred.append(tr.argmax(yhat.cpu(), dim=1))

            loss = self.loss_func(yhat, y)
            epoch_loss += loss.item()
            
        bal_acc = balanced_accuracy_score(tr.cat(ref), tr.cat(pred))

        return epoch_loss/len(loader), bal_acc
        
net = CNNAustral(nclasses=4, input_channels=3, device="cuda")
net


# ## Entrenamiento y validación
# Realizar el entrenamiento recorriendo varias veces el dataset (épocas). Se implementa una forma de early-stop: cuando el loss de validación no se mejora en una serie de épocas, se detiene el entrenamiento. 
# 
# **Nota**: Solo se guarda el modelo en los puntos donde la validación resulta ser mejor   

# In[42]:


log = []
best_loss, counter, patience = 999, 0, 3
for epoch in range(100):
    train_loss = net.fit(train_loader, verbose=True)
    val_loss, val_acc = net.test(val_loader)

    if val_loss < best_loss:    
        best_loss = val_loss
        tr.save(net.state_dict(), 'model.pmt')
        counter = 0
    else:
        counter += 1
        if counter > patience:
            break
            
    print(f'Epoch {epoch}, train_loss {train_loss:.2f}, val_loss {val_loss:.2f},  val_acc {val_acc:.2f}')
    log.append([train_loss, val_loss])
  


# ## Evaluación
# Tomamos la partición de test independiente para evaluar nuestro modelo

# In[ ]:


T = transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

########################################### Update data set

net = CNNAustral(nclasses=4, input_channels=3, device="cuda")

# carga de parámetros
net.load_state_dict(tr.load("model.pmt"))
net.eval()

test_loss, test_acc = net.test(test_loader)
   
print(f"Test loss {test_loss:.2f}, Test acc {test_acc:.2f}")


# ## Realizamos las predicciones en los datos nuevos

# In[ ]:


dataset_label = MeliDataset(csv_file = "to_label.csv",
                      root_dir = "imagenes/MLA1652",
                      transform = transforms.Compose(  
                           [transforms.ToPILImage(),
                            transforms.Resize((60,60)),
                            
                           transforms.ToTensor() ] )
                      )
#####################################

tolabel_loader = DataLoader(dataset_label, batch_size=BATCH_SIZE, shuffle=False)


# In[ ]:


########################################### Update data set

net = CNNAustral(nclasses=4, input_channels=3, device="cuda")

# carga de parámetros
net.load_state_dict(tr.load("model.pmt"))

net.eval()
from torch.autograd import Variable

x = Variable(torch.randn(60, D_in))


# In[ ]:




