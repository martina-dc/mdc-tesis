#!/usr/bin/env python
# coding: utf-8

# # Análisis de Datos Exploratorio
# Se levantará el dataset guardado en el notebook de preparación de datos, para poder analizarlo.

# In[1]:


import pandas as pd
import os
import json
import ast
import numpy as np
from pyprojroot import here
import sys
sys.path.append(here())
from utils.utils import normalizar_lineas_procesador, separar_valor_ram
    


# In[2]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt 


# In[4]:


path = here() / "data"


# In[5]:


df = pd.read_csv(path / "datos_laptops_transformed.csv", sep = ";")


# ## Preview del dataset

# In[6]:


df.head(3)


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.shape


# Haremos un analisis exploratorio con los siguientes campos para poder determinar cuales usar para las categorias:
# - precio
# - marca
# - resolucion
# - es gamer
# - velocidad max del procesador
# - marca
# - linea del procesador
# - memoria ram
# - capacidad del ssd
# - cuotas
# - tamaño de la pantalla  
# - costo de envío
# - peso

# ### Precio

# In[10]:


df.price.describe()


# In[11]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,5))
ax = sns.boxplot(x="price", data=df)
ax.set_title("Precio de la notebook", size=14, family='serif')  
plt.xlabel('Precio', size=12, family='serif')  
plt.show()


# In[12]:


df[df.price > 2500000]


# Tiene sentido esta computadora que es una MAC de alto nivel, por lo tanto no la eliminaremos del dataframe.
# 
# ### Marca  
# Como existen muchas marcas se procede a agrupar aquellas que aparezcan menos de 100 veces, bajo el nombre "Otros". Se entiende que las marcas más populares estan dentro de las que tienen mas de 100 y que no sería tan relevante dejar el nombre de una marca que aparece menos de 100 veces.

# In[13]:


marcas = df.Marca.value_counts().to_frame()
marcas["condicion"] = marcas.index
marcas.loc[marcas["Marca"] < 100, "condicion"] = "Otro"
marcas = marcas.rename_axis('Marca_Original').reset_index()
marcas.columns = ["Marca_Original", "Cantidad", "Marca_Nueva"]
marcas.tail()

del marcas["Cantidad"]


# In[14]:


df = df.merge(marcas,
         how = "left",
         left_on = "Marca",
         right_on = "Marca_Original")


# In[15]:


del df["Marca_Original"], df["Marca"]
df.rename(columns = {"Marca_Nueva" : "Marca"}, inplace = True)


# In[16]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(10,6))
ax = sns.boxplot(x="price", data=df, y = "Marca")  
ax.set_title("Precio de la notebook por marca", size=14, family='serif') 
plt.ylabel('Marca', size=12, family='serif')  
plt.xlabel('Precio', size=12, family='serif')  
plt.show()


# In[17]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,4))
ax = sns.boxplot(x="price", data=df[df["price"] < 1.000000e+07], y = "Es gamer")
ax.set_title("Gamer", size=14, family='serif') 
plt.ylabel('Gamer', size=12, family='serif')  
plt.xlabel('Precio', size=12, family='serif')  
plt.show()


# ## Procesador

# In[18]:


df = df.rename(columns= {"Línea del procesador" : "linea_procesador"})
df.linea_procesador.isna().sum()


# In[19]:


df = normalizar_lineas_procesador(df, "linea_procesador")
df.linea_procesador.isna().sum()


# In[20]:


df = df.rename(columns= {"Modelo del procesador" : "modelo_procesador"})
df.modelo_procesador.isna().sum()


# In[21]:


df = normalizar_lineas_procesador(df, "modelo_procesador")
df.modelo_procesador.isna().sum()


# In[22]:


df = normalizar_lineas_procesador(df, "modelo_procesador")
df.modelo_procesador.isna().sum()


# In[26]:


cm1 = (df.modelo_procesador == "M1")
cotro = (df.linea_procesador == "Otro")

df.loc[cm1 & cotro, "linea_procesador"] = "M1"


# In[28]:


df.linea_procesador.value_counts()


# In[27]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,4))
ax = sns.boxplot(x="price", data=df, y = "linea_procesador")
ax.set_title("Linea Procesador", size=14, family='serif') 
plt.ylabel('Linea Procesador', size=12, family='serif')  
plt.xlabel('Precio', size=12, family='serif')  
plt.show()


# In[21]:


df["Memoria RAM"].head()


# ### RAM

# In[23]:


df = separar_valor_ram(df,"Memoria RAM", 'Capacidad RAM', 'Medida_RAM')


# In[28]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,4))

plt.figure(figsize=(10,5))
ax = sns.boxplot(x="Capacidad RAM", data=df)
ax.set_title("Tamaño RAM (gb)", size=14, family='serif') 
plt.xlabel('Tamaño RAM (gb)', size=12, family='serif')  
plt.show()


# In[31]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,5))

ax = sns.boxplot(x="Capacidad RAM", data=df[df["Capacidad RAM"] < 129], y = "Marca")
ax.set_title("Tamaño RAM (gb) por Marca", size=14, family='serif') 
plt.xlabel('Tamaño RAM (gb)', size=12, family='serif')  
plt.ylabel('Marca', size=12, family='serif')  
plt.show()


# ### Peso
# 
# # NORMALIZAR EL PESO Y LUEGO SEGUIR CON EL RESTO DE LOS QUE PUSE EN EL CUESTIONARIO

# In[31]:


df.Peso.head()


# In[30]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,4))

plt.figure(figsize=(10,5))
ax = sns.boxplot(x="Peso de la notebook", data=df)
ax.set_title("Peso", size=14, family='serif') 
plt.xlabel('Tamaño RAM (gb)', size=12, family='serif')  
plt.show()


# In[ ]:




