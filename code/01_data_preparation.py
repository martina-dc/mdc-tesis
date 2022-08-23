#!/usr/bin/env python
# coding: utf-8

# ## Data Preparation

# In[1]:


import pandas as pd
import os
import json
import ast
import numpy as np
try:
    from pyprojroot import here
    print("Libreria: 'pyprojroot' esta instalada y se cargo correctamente")
except ModuleNotFoundError:
    print("Libreria: 'pyprojroot' no esta instalada, se debe instalar")
    get_ipython().system('pip install pyprojroot')
    
import sys
sys.path.append(here())
from utils.utils import get_attributes_from_row


# In[2]:


here()


# In[3]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# In[4]:


path = here() / "data"


# In[5]:


df = pd.read_csv(path / "datos_laptops.csv", sep = ';')


# In[6]:


df.head()


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df = df.drop_duplicates()


# In[10]:


df.shape


# # Seleccionamos las columnas de interes

# In[12]:


selected_cols = ["id", "title","seller_id","price","base_price","deal_ids","initial_quantity","sold_quantity", "listing_type_id",
                 "available_quantity","sold_quantity","sale_terms","condition", "installment", "display_size", "shipping_cost",
                 "descriptions","attributes","warnings","status","date_created","shipping.mode","shipping.free_shipping"]

df = df[selected_cols]


# In[13]:


df.status.value_counts()


# #### Filtramos aquellas que no esten activas

# In[14]:


c_active = df.status.isin(["active"])
df = df[c_active]


# In[15]:


df["shipping.mode"].value_counts()


# In[16]:


df["shipping.free_shipping"].value_counts()


# In[17]:


df["listing_type_id"].value_counts()


# #### Borramos del df original aquellas que tengan mas del 50% nulos

# In[18]:


perc = 50.0 # Like N %
min_count =  int(((100-perc)/100)*df.shape[0] + 1)
df = df.dropna( axis=1, 
                thresh=min_count)
df.info(verbose = True, show_counts  = True)


# ## Desarmamos las columnas attributes y sale_terms que son un dicc

# In[19]:


df_attributes = df[["id","attributes"]].copy()
dummy = df_attributes["attributes"].apply(lambda x: ast.literal_eval(x))

df_sale_terms = df[["id","sale_terms"]].copy()
dummy_st = df_sale_terms["sale_terms"].apply(lambda x: ast.literal_eval(x))


# In[20]:


lista_filas = []
lista_id = []
for (index_dummy, row), (index_df, rowa) in zip(pd.DataFrame(dummy).iterrows(), df_attributes.iterrows()):
    lista_filas.append(get_attributes_from_row(row))
    lista_id.append(rowa['id'])

lista_filas_st = []
lista_id_st = []
for (index_dummy, row), (index_df, rowa) in zip(pd.DataFrame(dummy_st).iterrows(), df_sale_terms.iterrows()):
    lista_filas_st.append(get_attributes_from_row(row))
    lista_id_st.append(rowa['id'])


# In[21]:


for df1, df2 in zip(lista_filas, lista_id):
    df1["id"] = df2

df_attributes = pd.concat(lista_filas)
display(df_attributes.head())


# #### Borramos aquellas columnas del df de atributtes que tengan mas del 50% de nulos

# In[22]:


perc = 50.0 # Like N %
min_count =  int(((100-perc)/100)*df_attributes.shape[0] + 1)
df_attributes = df_attributes.dropna( axis=1, 
                thresh=min_count)
df_attributes.info(verbose = True, null_counts = True)


# #### Garantia del producto

# In[23]:


for df3, df4 in zip(lista_filas_st, lista_id_st):
    df3["id"] = df4

df_st = pd.concat(lista_filas_st)
df_st.head()


# In[24]:


del df_st["Cantidad máxima de compra"], df_st["Disponibilidad de stock"], df_st["Facturación"]


# In[25]:


df_st.head()


# ### Juntamos todos los dataframes

# In[26]:


display(df.columns)
display(df.shape)
display(df.id.nunique())


# In[33]:


df_attributes.drop_duplicates(inplace = True)
display(df_attributes.columns)
display(df_attributes.shape)
display(df_attributes.id.nunique())


# In[34]:


df_st.drop_duplicates(inplace = True)
display(df_st.columns)
display(df_st.shape)
display(df_st.id.nunique())


# #### Merge

# In[37]:


display(df.shape)
df = df.merge(df_attributes, on = "id", how = "left")
display(df.shape)


# In[38]:


display(df.shape)
df = df.merge(df_st, on = "id", how = "left")
display(df_.shape)


# In[39]:


df.head()


# In[42]:


df.to_csv(path / "datos_laptops_transformed.csv", index = False, sep = ";")

