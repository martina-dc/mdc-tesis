#!/usr/bin/env python
# coding: utf-8

# # 00 - Extraccion de datos mediante API de Mercado Libre

# ## Importación de librerias

# In[1]:


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


# In[2]:


here()


# In[3]:


path_save = here() / "data" 


# In[4]:


import sys
sys.path.append(here())
from utils.utils import get_q_items, create_item_list, get_available_filters, get_df_list


# ## Armado del Dataset

# Para armar el dataset descargaremos para Mercado Libre Argentina laptops publicadas. La categoría se llama: MLA1652.
# La API tiene una restricción que no permite utilizar un offset mayor a 4000. Esto nos permite descargar menos de 4000 publicaciones por loop.
# Es por ello que se han armado ciertos grupos para descargar la información por grupo. Esto nos permite tambien etiquetar a las publicaciones con este atributo, 
# elemento que no se obtiene descargando la información de la publicación.

# <img src="Grupos.png" alt="Drawing" style="width: 600px;"/>
# 

# Para armar estos grupos necesitamos filtar desde el url mediante los siguientes nombres:  
# 
# * installments:  
#     + yes  
#     + no_interest  
# * power_seller:  
#     + yes  
#     + no  
# * shipping_cost:  
#     + get_available_filtersfree  
#             

# ### Grupo 1

# In[5]:


filters = {"installments":"yes",
           "display_size":'(*-14.1")',
          "shipping_cost":"free"}


# In[6]:


maximum = get_q_items(filters = filters)
print(f'Encontramos {maximum} resultados para nuestra consulta')


# In[7]:


item_list = create_item_list(filters = filters)


# In[8]:


data = get_df_list(item_list)
data["installment"] = "yes"
data["display_size"] = "hasta 14.1" 
data["shipping_cost"] = "free"


# ### Grupo 2

# In[9]:


filters = {"installments":"yes",
           "display_size":'[14.1"-17")',
                "shipping_cost":"free"}


# In[10]:


maximum = get_q_items(filters = filters)
print(f'Encontramos {maximum} resultados para nuestra consulta')


# In[11]:


item_list = create_item_list(filters = filters)


# In[12]:


data2 = get_df_list(item_list)
data2["installment"] = "yes"
data2["display_size"] = "entre 14.1 y 16.9"
data2["shipping_cost"] = "free"


# ### Grupo 3

# In[13]:


filters = {"installments":"no_interest",
           "display_size":'(*-14.1")',
          "shipping_cost":"free"}


# In[14]:


maximum = get_q_items(filters = filters)
print(f'Encontramos {maximum} resultados para nuestra consulta')


# In[15]:


item_list = create_item_list(filters = filters)


# In[16]:


data3 = get_df_list(item_list)
data3["installment"] = "no_interest"
data3["display_size"] = "hasta 14.1" 
data3["shipping_cost"] = "free"


# ### Grupo 4

# In[17]:


filters = {"installments":"no_interest",
           "display_size":'[14.1"-17")',
          "shipping_cost":"free"}


# In[18]:


maximum = get_q_items(filters = filters)
print(f'Encontramos {maximum} resultados para nuestra consulta')


# In[19]:


item_list = create_item_list(filters = filters)


# In[20]:


data4 = get_df_list(item_list)
data4["installment"] = "no_interest"
data4["display_size"] = "entre 14.1 y 16.9"
data4["shipping_cost"] = "free"


# ________________________

# In[21]:


data = pd.concat([data, data2, data3, data4])


# In[22]:


data.shape


# In[23]:


data.title.value_counts()


# In[24]:


data.to_csv(path_save / "datos_laptops.csv", index = False, sep = ";")

