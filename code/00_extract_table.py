#!/usr/bin/env python
# coding: utf-8

# # 00 - Extraccion de datos mediante API de Mercado Libre

# ## Importación de librerias

# In[43]:


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
    get_ipython().system('pip install pyprojroot')

import matplotlib.pyplot as plt
import os

try:
    import seaborn as sns
    print("Libreria: 'seaborn' esta instalada y se cargo correctamente")
except ModuleNotFoundError:
    print("Libreria: 'seaborn' no esta instalada, se debe instalar")
    get_ipython().system('pip install seaborn')


# In[44]:


from google.colab import drive
drive.mount('/content/drive')


# In[45]:


path_secrets = "/content/drive/MyDrive/Austral/Maestria/Tesis/Codigo/secrets/"


# ## Definicion de funciones a utilizar para obtener los datos

# In[46]:


def auth_token():
    with open(path_secrets + 'secrets.json') as f:
        secrets = json.load(f)
    
    print(f'http://auth.mercadolibre.com.ar/authorization?response_type=code&client_id={secrets["client_id"]}&redirect_uri={secrets["redirect_uri"]}')
    codigo = input('Codigo :')
    url = 'https://api.mercadolibre.com/oauth/token'
    response = requests.post(url, data = {
        
        'grant_type' : 'authorization_code',
        'client_id': secrets['client_id'],
        'client_secret': secrets['client_secret'],
         'code' : codigo,
        'redirect_uri': secrets['redirect_uri']
    })
    print(response)
    return response.json()['access_token']

meli_auth_token = None

def get_response(url):
    global meli_auth_token
    
    try:
        if meli_auth_token is None and 'mercadolibre' in url:
            meli_auth_token = auth_token()
        response = requests.get(url, timeout = 4,
                                   headers={'Authorization': f'Bearer {meli_auth_token}'})
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as errh:
        print ("Http Error:",errh)
    except requests.exceptions.ConnectionError as errc:
        print ("Error Connecting:",errc)
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)
    except requests.exceptions.RequestException as err:
        print ("OOps: Something Else",err)
        
#Definimos una función para traer el elemento que queremos
def traerdato(elemento,rama,subrama,valor='value_name'):
    indices=[]
    for i,s in enumerate(elemento[rama]):
        for j in s:
            if subrama in str(s[j]):
                indices.append([i,s])
    if len(indices) == 0:
        return 'Sin Datos'
    else:
        return indices[0][1][valor]


# In[47]:


meli_auth_token = auth_token()


# ## Armado del Dataset

# Para armar el dataset descargaremos para Mercado Libre Argentina laptops publicadas. La categoría se llama: MLA1652

# In[48]:


url='https://api.mercadolibre.com/sites/MLA/search?category=MLA1652&search_type=scan#json'

r = get_response(url) 
maximum = int(str(r.json()["paging"]["total"])) #Guardamos el número máximo de resultados para luego stoppear nuestro loop

print(f'Encontramos {maximum} resultados para nuestra consulta')


# In[49]:


offset_max = maximum / 50
offset_max


# In[50]:


offset = 0
item_list = []

while r.status_code == 200 and offset <= round(offset_max):
    url=f'https://api.mercadolibre.com/sites/MLA/search?category=MLA1652&offset={offset}#json'
    r = get_response(url)
    if r is not None:
      data = r.json()
      length = len(data['results'])
      for i in range(length):
          item_id = data['results'][i]['id']
          item_list.append(item_id)
      print("Porcentaje de completitud: {:0.2%}".format(offset/maximum),end='\r')

    offset += 1
          
len(item_list)


# In[52]:


final_list = []
for i in range(len(item_list)):
    item="https://api.mercadolibre.com/items/{}".format(item_list[i])
    item_add = requests.get(item)
    item_add = item_add.json()
    final_list.append(item_add)
    print("Porcentaje de completitud: {:0.2%}".format((i+1)/len(item_list)),end='\r')


# In[ ]:


print(len(item_list))
print(len(set(item_list)))


# In[56]:


data = pd.json_normalize(final_list)
data.head(3)


# In[57]:


data.title.value_counts()


# In[60]:


path_save = "/content/drive/MyDrive/Austral/Maestria/Tesis/Data/"


# In[61]:


data.to_csv(path_save + "datos_laptops.csv", index = False, sep = ";")


# In[ ]:




