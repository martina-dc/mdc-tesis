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

# In[2]:


import pandas as pd
import os
import json
import ast
import numpy as np


# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# In[21]:


path = "/content/drive/MyDrive/Austral/Maestria/Tesis/Data"


# In[113]:


df = pd.read_csv(path + "datos_laptops.csv",  sep = ";")


# ## Etiquetado 

# In[80]:


df["Es gamer"].value_counts()


# In[81]:


df["velocidad_max_procesador"].isna().sum()


# In[82]:


cond_gamer = (df["Es gamer"] == "Sí")
cond_mac_MSI = df["Marca"].isin(["MSI","Apple"])
cond_ram = df["Capacidad RAM"] > 12
cond_procesador = df["linea_procesador"].isin(["Core i7", "Core i9", "Ryzen 7", "Ryzen 9"])


# In[83]:


df["label"] = "Estandar Tradicional"
df.loc[cond_gamer, "label"] = "Gamer"
df.loc[df["Es gamer"].isna(), "label"] = ""
df.loc[cond_mac_MSI, "label"] = "Estandar Premium"
df.loc[cond_ram, "label"] = "Estandar Premium"
df.loc[df["Capacidad RAM"] == -1, "label"] = ""
df.loc[df["linea_procesador"].isna(), "label"] = ""
df.loc[cond_procesador, "label"] = "Estandar Premium"


# In[84]:


df.label.value_counts()


# In[85]:


df.head()


# In[86]:


to_label = ( df.label == "")
to_train = ( ( df.label != "") & ( df.label.notna()))


# In[87]:


df[to_label].shape


# In[88]:


df[to_train].shape


# In[89]:


df[to_label].to_csv("/content/drive/MyDrive/Austral/Maestria/Redes/Trabajo Final Redes/ML_dataset/to_label.csv", index = False)
df[to_train].to_csv("/content/drive/MyDrive/Austral/Maestria/Redes/Trabajo Final Redes/ML_dataset/to_train.csv", index = False)


# In[90]:


labels = df[["id", "label"]]
labels.head()
path_images = "/content/drive/MyDrive/Austral/Maestria/Redes/Trabajo Final Redes/imagenes/MLA1652/"
labels["path"] = path_images + "categ-MLA1652_itemid-" + df.id.copy() + ".jpg"


# In[91]:


lista = []
for file in os.listdir(path_images):
  lista.append(file)


# In[92]:


df_fotos = pd.DataFrame(lista)
df_fotos.columns = ["0"]
df_fotos["path_real"] = path_images + + df_fotos["0"].copy()
del df_fotos["0"]
df_fotos.head()


# In[93]:


labels.head()


# In[94]:


labels.shape


# In[95]:


labels.path.nunique()


# In[96]:


df_fotos.shape


# In[97]:


df_fotos.path_real.nunique()


# In[98]:


result = labels.merge(df_fotos,
             how = "left",
             left_on = "path",
             right_on = "path_real")
del result["path"], result["id"]


# In[99]:


result.head()


# In[100]:


result["path_real"] = result["path_real"].str.split("/").str[10]
result["dummy"] = 1


# In[101]:


result.head()


# In[102]:


del result["dummy"]



# {"Estandar Tradicional" : 1,
#                "Estandar Premium" : 2,
#                "Gamer" : 3}
#                

# In[103]:


result["numeric_label"] = np.where(result["label"] == "Estandar Tradicional", 1 ,
                                   np.where(result["label"] == "Estandar Premium" ,2 ,
                                            np.where(result["label"] == "Gamer" ,3 , np.NaN)
                                  ))
                                  
del result["label"]


# In[104]:


(result.numeric_label == 1).sum()


# In[105]:


result.numeric_label.isna().sum()


# In[106]:


to_label_new = ( (result.numeric_label.isna()) & (result.path_real.notna()) )
to_train_new = ( (result.numeric_label != "") & (result.numeric_label.notna())  & (result.path_real.notna()) & (result.path_real != ""))


# In[107]:


result[to_label_new].head()


# In[108]:


result[to_label_new].shape


# In[109]:


result[to_train_new].head()


# In[110]:


result[to_train_new].shape


# In[111]:


result[to_train_new].to_csv("/content/drive/MyDrive/Austral/Maestria/Redes/Trabajo Final Redes/to_train.csv", index = False)
result[to_label_new].to_csv(path + "to_label.csv", index = False)

