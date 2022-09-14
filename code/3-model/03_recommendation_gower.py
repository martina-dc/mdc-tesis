#!/usr/bin/env python
# coding: utf-8

# 
# # Data labeling  
# En este notebook lo que haremos es por un lado tomar el dataset que contiene las 10000 laptops descargadas y comprender las columnas que obtuvimos.
# Por otro lado lo que haremos es cargar en una columna nueva un path a donde se encuentran ubicadas las imagenes de este laptop. 
# Una vez que entendamos cuantas laptops con fotos tenemos y cuantas pudimos etiquetar, separaremos el dataset en 3. Una parte para entrenamiento, otra para validacion y otra para test. Aquellas que quedaron sin etiqueta son las que querremos etiquetar para poder incorporarlas a nuestra database de donde sacaremos las recomendaciones a los clientes.
# 
# El etiquetado buscara clasificar las notebooks en 3 tipos:
# 
# - Tradicional economica
# - Tradicional premium
# - Gamers

# In[1]:


import pandas as pd
import os
import json
import ast
import numpy as np
from pyprojroot import here

# In[2]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# In[3]:


path = here() / "data"

# In[4]:


file = input("Ingrese el nombre del file que desea leer: \n")

# In[5]:


df = pd.read_csv(path / file,  sep = ";")

# In[6]:


df.head()

# In[7]:


df.shape

# ### Cargamos el dataset con todas las combinaciones de respuestas posibles y una notebook etiquetada

# In[8]:


respuestas = pd.read_csv(path / "respuestas_posibles.csv",  sep = ",",  encoding='windows-1251')

# In[9]:


respuestas.shape

# In[10]:


respuestas.head()

# In[11]:


del respuestas["cuotas"]

# In[12]:


respuestas.shape

# In[13]:


respuestas.groupby(["gasto", "traslado", "uso", "teclado numerico"]).size().reset_index(name = "q").sort_values(by = "q", ascending= False).head()

# In[14]:


re = respuestas.merge(respuestas, on = ["gasto", "traslado", "uso", "teclado numerico"]).merge(respuestas, on = ["gasto", "traslado", "uso", "teclado numerico"])
re.drop_duplicates(inplace= True)

# In[15]:


re.head()

# ### Importamos las distancias de Gower entre las PCs

# In[16]:


file2 = input("Ingrese el archivo que tiene los valores de Gower que quiere leer: \n")

# In[17]:


df_gower = pd.read_csv(path / file2,  sep = ";")

# In[18]:


df_gower.head()

# In[19]:


len(df.id.to_list())

# In[20]:


df_gower.columns = df.id.to_list()
df_gower.index = df.id.to_list()

# In[23]:


dic = dict()
for col in df_gower.columns:
   # dic[col] = df_gower[col].sort_values().head(4).values
    dic[col] = df_gower[col].sort_values().head(11).index

    

# In[24]:


dic

# ### Facilidades de Pago

# In[25]:


file3 = input("Ingrese el nombre del archivo que tiene la infromación de las notebooks que tienen cuotas: \n")

# In[27]:


df_facilidades = pd.read_csv(path / file3, sep = ";")

# In[28]:


df_facilidades.head()

# 
# ## Recommendation

# 

# In[ ]:



