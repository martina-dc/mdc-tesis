#!/usr/bin/env python
# coding: utf-8

# # 00 - Extraccion de datos mediante API de Mercado Libre

# ## Importación de librerias

# In[46]:


import pandas as pd
import numpy as np
import json
import requests

from datetime import datetime, timedelta, timezone, date
pd.options.display.max_columns=None

try:
    from pyprojroot import here
    print("Libreria: 'pyprojroot' esta instalada y se cargo correctamente")
except ModuleNotFoundError:
    print("Libreria: 'pyprojroot' no esta instalada, se debe instalar")
    !pip install pyprojroot

import matplotlib.pyplot as plt
import os

import urllib.request
from PIL import Image

import seaborn as sns

try:
    import seaborn as sns
    print("Libreria: 'seaborn' esta instalada y se cargo correctamente")
except ModuleNotFoundError:
    print("Libreria: 'seaborn' no esta instalada, se debe instalar")
    !pip install seaborn


# In[47]:


path_save = here() / "data" 
path_secrets = here() / "secrets"   
drive.mount('/content/drive')

# In[48]:


import sys
sys.path.append(here())
from utils.utils import get_q_items, create_item_list, get_available_filters, get_response

# ## Definicion de funciones a utilizar para obtener los datos

# ## Armado del Dataset

# Para armar el dataset descargaremos para Mercado Libre Argentina laptops publicadas. La categoría se llama: MLA1652

# In[51]:


url='https://api.mercadolibre.com/sites/MLA/search?category=MLA1652&search_type=scan#json'

r = get_response(url) 
maximum = int(str(r.json()["paging"]["total"])) #Guardamos el número máximo de resultados para luego stoppear nuestro loop

print(f'Encontramos {maximum} resultados para nuestra consulta')

# In[52]:


offset = 0
item_list = []

# Definimos una función para traer los items de una búsqueda cambiando la categoria
def buscaritems(categ = "MLA1652" ,offset=0, maximum=None):
    # Primero indicamos la consulta que vamos a hacer
    url=f'https://api.mercadolibre.com/sites/MLA/search?category={categ}&offset=0#json'
    r = get_response(url) #Vemos los resultados de la consulta
    if maximum==None:
        #Consultamos cuál es el número máximo de resultados
        maximum = int(str(r.json()["paging"]["total"]))
    print(f'Encontramos {maximum} resultados para nuestra consulta')
    # Vamos a forzarnos a buscar pocos resultados para seguir testeando y no perder tanta velocidad
    # Cuando vayamos full mode, esta linea de abajo hay que comentarla o borrarla
    # Ahora, sabiendo esto, vamos a traer todos los resultados iterando y aumentando el offset en cada loop.
    # Recuerden que solo podemos traer de a tandas de a 50 resultados
    item_list = []
    categories_list=[]
    while r.status_code == 200 and offset <= maximum:
        url=f'https://api.mercadolibre.com/sites/MLA/search?category={categ}&offset={offset}#json'
        r = get_response(url)
        data = r.json()
        length = len(data['results'])
        for i in range(length):
            item_id = data['results'][i]['id']
            item_list.append(item_id)
            categories_list.append(categ)
        offset += 50
    return item_list,categories_list

# In[53]:


categorias = [{"id": "MLA1652", "name": "Laptops"}]

# In[54]:


#limite de cantidad de imagenes
maximum = 10000

item_list=[]
categories_list=[]
for i in categorias:
    print(f"Buscamos items de la categoría {i['name']}")
    temp_items,temp_categories=buscaritems(i['id'],0,maximum=maximum)
    item_list=item_list+temp_items
    categories_list=categories_list+temp_categories

# In[55]:


item=[]
for j,i in enumerate(item_list):
    url=f'https://api.mercadolibre.com/items/{i}'
    item.append(get_response(url).json())
    print("Porcentaje de completitud: {:0.2%}".format((j+1)/len(item_list)),end='\r')

# In[56]:


path = "/content/drive/MyDrive/Austral/Maestria/Redes/Trabajo Final/"

# In[57]:


def crearcarpeta(carpeta):
  if not os.path.exists(path + carpeta):
    os.makedirs(path + carpeta)

# In[58]:


for z,i in enumerate(item):
  crearcarpeta(f"imagenes/{categories_list[z]}")
  urllib.request.urlretrieve(i['pictures'][0]['secure_url'], path + f"imagenes/{categories_list[z]}/categ-{i['category_id']}_itemid-{i['id']}.jpg")

# In[58]:



