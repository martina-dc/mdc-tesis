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


# ## Preview del dataset

# In[114]:


df.head(3)


# In[115]:


df = df[["id", "title", "price", "base_price", "sale_terms", "permalink", "pictures", "attributes"]]
df.head()


# In[116]:


df.shape


# In[ ]:


df_attributes = df[["id","attributes"]].copy()
dummy = df_attributes["attributes"].apply(lambda x: ast.literal_eval(x))


# In[ ]:


def get_attributes_from_row(row):
  fila = dict()
  for e in row:
    for element in e:
      #print(element)
      fila[str(element['name'])] =  element['value_name']
    
  return pd.DataFrame.from_dict(fila, orient='index').transpose()


# In[28]:


lista_filas = []
lista_id = []
for (index_dummy, row), (index_df, rowa) in zip(pd.DataFrame(dummy).iterrows(), df_attributes.iterrows()):
  lista_filas.append(get_attributes_from_row(row))
  lista_id.append(rowa['id'])
    


# In[29]:


for df1, df2 in zip(lista_filas, lista_id):
  df1["id"] = df2

df_attributes = pd.concat(lista_filas)
df_attributes.head()


# In[30]:


df = df.drop_duplicates()
df_attributes = df_attributes.drop_duplicates()


# In[31]:


df.shape[0] == df.id.nunique()


# In[32]:


df_attributes.shape[0] == df_attributes.id.nunique()


# In[33]:


df = df.merge(df_attributes,
               how = "left",
               on = "id")
df.head()


# In[34]:


if 'attributes' in df.columns.values:
  del df['attributes']

if 'pictures' in df.columns.values:
  del df['pictures'], 

if 'sale_terms' in df.columns.values:
  del df['sale_terms']


# In[35]:


df.head()


# In[36]:


df.info(verbose=True, null_counts=True)


# In[37]:


df.shape


# In[38]:


df = df[["id", "price", "Marca", "Resolución de la pantalla", "Es gamer", "Velocidad máxima del procesador",
         "Capacidad del SSD", "Memoria RAM", "Velocidad de la memoria RAM", "Línea del procesador", "Marca del procesador",
         "Capacidad del SSD", "Frecuencia de actualización de la pantalla"]]


# Haremos un analisis exploratorio con los siguientes campos para poder determinar cuales usar para las categorias:
# - price
# - marca
# - resolucion
# - es gamer
# - velocidad max del procesador
# - marca
# - linea del procesador
# - memoria ram
# - capacidad del ssd

# In[39]:


import seaborn as sns
import matplotlib.pyplot as plt 


# In[40]:


df.price.describe()


# In[41]:


sns.set_theme(style="whitegrid")

plt.figure(figsize=(12,7))
ax = sns.boxplot(x="price", data=df[df["price"] < 1.000000e+07])
ax.set_title("Laptop's Price")
plt.show()


# In[42]:


marcas = df.Marca.value_counts().to_frame()
marcas["condicion"] = marcas.index
marcas.loc[marcas["Marca"] < 100, "condicion"] = "Otro"
marcas = marcas.rename_axis('Marca_Original').reset_index()
marcas.columns = ["Marca_Original", "Cantidad", "Marca_Nueva"]
marcas.tail()

del marcas["Cantidad"]


# In[43]:


df = df.merge(marcas,
         how = "left",
         left_on = "Marca",
         right_on = "Marca_Original")


# In[44]:


df.head()


# In[45]:


del df["Marca_Original"], df["Marca"]
df.rename(columns = {"Marca_Nueva" : "Marca"}, inplace = True)


# In[46]:


df.head()


# In[47]:


sns.set_theme(style="whitegrid")

plt.figure(figsize=(12,7))
ax = sns.boxplot(x="price", data=df[df["price"] < 1.000000e+07], y = "Marca")
ax.set_title("Laptop's Price")
plt.show()


# In[48]:


plt.figure(figsize=(12,7))
ax = sns.boxplot(x="price", data=df[df["price"] < 1.000000e+07], y = "Es gamer")
ax.set_title("Laptop's Price")
plt.show()


# ## Procesador

# In[49]:


df["Línea del procesador"].value_counts().to_frame().head()


# In[50]:


df["Línea del procesador"].nunique()


# In[51]:


condic_i3 = df["Línea del procesador"].str.lower().str.contains("i3", na=False)
condic_i5 = df["Línea del procesador"].str.lower().str.contains("i5", na=False)
condic_i7 = df["Línea del procesador"].str.lower().str.contains("i7", na=False)
condic_i9 = df["Línea del procesador"].str.lower().str.contains("i9", na=False)

df["linea_procesador"] = df["Línea del procesador"].copy()
df.loc[condic_i3, "linea_procesador"] = "Core i3"
df.loc[condic_i5, "linea_procesador"] = "Core i5"
df.loc[condic_i7, "linea_procesador"] = "Core i7"
df.loc[condic_i9, "linea_procesador"] = "Core i9"


# In[52]:


cond_ryzen = df["Línea del procesador"].str.lower().str.contains("ryzen", na=False)
condic_r3 = df["Línea del procesador"].str.lower().str.contains("3", na=False)
condic_r5 = df["Línea del procesador"].str.lower().str.contains("5", na=False)
condic_r7 = df["Línea del procesador"].str.lower().str.contains("7", na=False)
condic_r9 = df["Línea del procesador"].str.lower().str.contains("9", na=False)


df.loc[condic_r3 & cond_ryzen, "linea_procesador"] = "Ryzen 3"
df.loc[condic_r5& cond_ryzen, "linea_procesador"] = "Ryzen 5"
df.loc[condic_r7& cond_ryzen, "linea_procesador"] = "Ryzen 7"
df.loc[condic_r9& cond_ryzen, "linea_procesador"] = "Ryzen 9"


# In[53]:


condic_a6 = df["Línea del procesador"].str.lower().str.contains("a6", na=False)
condic_a8 = df["Línea del procesador"].str.lower().str.contains("a8", na=False)
condic_a10 = df["Línea del procesador"].str.lower().str.contains("a10", na=False)
condic_a12 = df["Línea del procesador"].str.lower().str.contains("a12", na=False)


df.loc[condic_a6, "linea_procesador"] = "AMD A6"
df.loc[condic_a8, "linea_procesador"] = "AMD A8"
df.loc[condic_a10, "linea_procesador"] = "AMD A10"
df.loc[condic_a12, "linea_procesador"] = "AMD A12"


# In[54]:


cond_celeron = df["Línea del procesador"].str.lower().str.contains("celeron", na=False)
df.loc[cond_celeron, "linea_procesador"] = "Celeron"


# In[55]:


cond_pentium = df["Línea del procesador"].str.lower().str.contains("pentium", na=False)
df.loc[cond_pentium, "linea_procesador"] = "Pentium"


# In[56]:


cond_athlon = df["Línea del procesador"].str.lower().str.contains("athlon", na=False)
df.loc[cond_athlon, "linea_procesador"] = "Athlon"


# In[57]:


cond_sempron = df["Línea del procesador"].str.lower().str.contains("sempron", na=False)
df.loc[cond_sempron, "linea_procesador"] = "Sempron"


# In[58]:


cond_m1 = df["Línea del procesador"].str.lower().str.contains("m1", na=False)
df.loc[cond_m1, "linea_procesador"] = "M1"

cond_m2 = df["Línea del procesador"].str.lower().str.contains("m2", na=False)
df.loc[cond_m2, "linea_procesador"] = "M2"


# In[59]:


df["linea_procesador"].nunique()


# In[60]:


proc = df.linea_procesador.value_counts().to_frame()
proc["condicion"] = proc.index
proc.loc[proc["linea_procesador"] < 9, "condicion"] = "Otro"
proc = proc.rename_axis('linea_procesador_Original').reset_index()
proc.columns = ["linea_procesador_Original", "Cantidad", "linea_procesador_Nueva"]
proc.tail()


# In[61]:


del proc["Cantidad"]


# In[62]:


df = df.merge(proc,
         how = "left",
         left_on = "linea_procesador",
         right_on = "linea_procesador_Original")
df.head()


# In[63]:


del df["linea_procesador"], df["linea_procesador_Original"], df["Línea del procesador"]

df = df.rename(columns = {"linea_procesador_Nueva"  : "linea_procesador"})
df.head()


# In[64]:


df[['Capacidad RAM', 'Medida_RAM']] = df["Memoria RAM"].str.split(' ', 1, expand=True)


# In[65]:


cond_ram_mb = (df["Medida_RAM"].str.lower() == "mb")
df.loc[cond_ram_mb, "Capacidad RAM"] = df['Capacidad RAM'].fillna(-1).astype(float, errors = "ignore").astype(int) / 1000
df.loc[cond_ram_mb, "Medida_RAM"] = "GB"


# In[66]:


cond_ram_kb = (df["Medida_RAM"].str.lower() == "kb")
df.loc[cond_ram_kb, "Capacidad RAM"] = df['Capacidad RAM'].fillna(-1).astype(float, errors = "ignore").astype(int) / 1000000
df.loc[cond_ram_kb, "Medida_RAM"] = "GB"


# In[67]:


df['Medida_RAM'].value_counts()


# In[68]:


df["Capacidad RAM"] = df["Capacidad RAM"].fillna(-1).astype(float)


# In[69]:


sns.set_theme(style="whitegrid")

plt.figure(figsize=(12,7))
ax = sns.boxplot(x="Capacidad RAM", data=df[df["Capacidad RAM"] < 129])
ax.set_title("Laptop's RAM")
plt.show()


# In[70]:


df["Capacidad RAM"].describe()


# In[71]:


sns.set_theme(style="whitegrid")

plt.figure(figsize=(12,7))
ax = sns.boxplot(x="Capacidad RAM", data=df[df["Capacidad RAM"] < 129], y = "Marca")
ax.set_title("Laptop's RAM")
plt.show()


# ## Velocidad max del procesador

# In[72]:


df["Velocidad máxima del procesador"].value_counts()


# In[73]:


df[['velocidad_max_procesador', 'medida_vel_procesador']] = df["Velocidad máxima del procesador"].str.split(' ', 1, expand=True)


# In[74]:


df.medida_vel_procesador.value_counts()


# In[75]:


cond_ram_MHz = (df["medida_vel_procesador"].str.lower() == "mhz")
df.loc[cond_ram_MHz, "velocidad_max_procesador"] = df['velocidad_max_procesador'].fillna(-1).astype(float, errors = "ignore").astype(int) / 1000
df.loc[cond_ram_MHz, "medida_vel_procesador"] = "GHz"


# In[76]:


cond_ram_Hz = (df["medida_vel_procesador"].str.lower() == "hz")
df.loc[cond_ram_Hz, "velocidad_max_procesador"] = df['velocidad_max_procesador'].fillna(-1).astype(float, errors = "ignore").astype(int) / 1000000000
df.loc[cond_ram_Hz, "medida_vel_procesador"] = "GHz"


# In[77]:


df["velocidad_max_procesador"] = df["velocidad_max_procesador"].astype(float)


# In[78]:


sns.set_theme(style="whitegrid")

plt.figure(figsize=(12,7))
ax = sns.boxplot(x="velocidad_max_procesador", data=df[df["velocidad_max_procesador"] < 40])
ax.set_title("Laptop's Vel procesador")
plt.show()


# In[79]:


df["velocidad_max_procesador"].describe()


# ##Etiquetado 

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

