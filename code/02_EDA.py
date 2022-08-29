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
from utils.utils import normalizar_lineas_procesador, separar_valor_um
    

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


file = input("Ingrese el nombre del archivo que desea leer: \n")

# In[6]:


df = pd.read_csv(path / file, sep = ";")

# ## Preview del dataset

# In[7]:


df.head(3)

# In[8]:


df.shape

# In[9]:


df.info()

# In[10]:


df.shape

# ## EDA
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

# In[11]:


df.price.describe()

# In[12]:


df = df[df.price > 50000]

# In[13]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,5))
ax = sns.boxplot(x="price", data=df)
ax.set_title("Precio de la notebook", size=14, family='serif')  
plt.xlabel('Precio', size=12, family='serif')  
plt.show()

# In[14]:


df[df.price > 2500000]

# Tiene sentido esta computadora que es una MAC de alto nivel, por lo tanto no la eliminaremos del dataframe.
# 
# ### Marca  
# Como existen muchas marcas se procede a agrupar aquellas que aparezcan menos de 100 veces, bajo el nombre "Otros". Se entiende que las marcas más populares estan dentro de las que tienen mas de 100 y que no sería tan relevante dejar el nombre de una marca que aparece menos de 100 veces.

# In[15]:


marcas = df.Marca.value_counts().to_frame()
marcas["condicion"] = marcas.index
marcas.loc[marcas["Marca"] < 100, "condicion"] = "Otro"
marcas = marcas.rename_axis('Marca_Original').reset_index()
marcas.columns = ["Marca_Original", "Cantidad", "Marca_Nueva"]
marcas.tail()

del marcas["Cantidad"]

# In[16]:


df = df.merge(marcas,
         how = "left",
         left_on = "Marca",
         right_on = "Marca_Original")


# In[17]:


del df["Marca_Original"], df["Marca"]
df.rename(columns = {"Marca_Nueva" : "Marca"}, inplace = True)

# In[18]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(10,6))
ax = sns.boxplot(x="price", data=df, y = "Marca")  
ax.set_title("Precio de la notebook por marca", size=14, family='serif') 
plt.ylabel('Marca', size=12, family='serif')  
plt.xlabel('Precio', size=12, family='serif')  
plt.show()

# In[19]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,4))

plt.figure(figsize=(10,5))
ax = sns.barplot(x="Marca", y = "total", data=df.groupby("Marca").size().reset_index(name="total"))
ax.set_title("Marca", size=14, family='serif') 
plt.xlabel("Tipo", size=12, family='serif')  
plt.ylabel("Cantidad PCs", size=12, family='serif')  
plt.show()

# ### Procesador

# In[20]:


df = df.rename(columns= {"Línea del procesador" : "linea_procesador"})
df.linea_procesador.isna().sum()

# In[21]:


df = normalizar_lineas_procesador(df, "linea_procesador")
df.linea_procesador.isna().sum()

# In[22]:


df = df.rename(columns= {"Modelo del procesador" : "modelo_procesador"})
df.modelo_procesador.isna().sum()

# In[23]:


df = normalizar_lineas_procesador(df, "modelo_procesador")
df.modelo_procesador.isna().sum()

# In[24]:


cm1 = (df.modelo_procesador == "M1")
cotro = (df.linea_procesador == "Otro")

df.loc[cm1 & cotro, "linea_procesador"] = "M1"

# In[25]:


df.linea_procesador.value_counts()

# In[26]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,5))
ax = sns.boxplot(x="price", data=df, y = "linea_procesador")
ax.set_title("Linea Procesador", size=14, family='serif') 
plt.ylabel('Linea Procesador', size=12, family='serif')  
plt.xlabel('Precio', size=12, family='serif')  
plt.show()

# ### RAM

# In[27]:


df = separar_valor_um(df = df,
                    colname = "Memoria RAM", 
                    res_val = 'Capacidad RAM', 
                    res_um = 'Medida_RAM',
                    cambio = {"mb" : 1000,
                              "kb" : 1000000},
                    new_um = "GB")

# In[28]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,4))

plt.figure(figsize=(10,5))
ax = sns.boxplot(x="Capacidad RAM", data=df)
ax.set_title("Tamaño RAM (gb)", size=14, family='serif') 
plt.xlabel('Tamaño RAM (gb)', size=12, family='serif')  
plt.show()

# In[29]:


df = df[df["Capacidad RAM"] < 129]

# In[30]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,5))

ax = sns.boxplot(x="Capacidad RAM", data=df, y = "Marca")
ax.set_title("Tamaño RAM (gb) por Marca", size=14, family='serif') 
plt.xlabel('Tamaño RAM (gb)', size=12, family='serif')  
plt.ylabel('Marca', size=12, family='serif')  
plt.show()

# ### Peso

# In[31]:


df = separar_valor_um(df = df,
                    colname = "Peso", 
                    res_val = 'Valor_Peso', 
                    res_um = 'Medida_Peso',
                    cambio = {"g" : 1000,
                              "lb" : 2.205},
                    new_um = "kg")

# In[32]:


c_peso_ok = (df["Valor_Peso"] < 5) & (df["Valor_Peso"] > 0)
df.loc[~c_peso_ok, "Valor_Peso"] = np.NaN
df.loc[~c_peso_ok, "Medida_Peso"] = np.NaN

# In[33]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,4))

plt.figure(figsize=(10,5))
ax = sns.boxplot(x="Valor_Peso", data=df)
ax.set_title("Peso", size=14, family='serif') 
plt.xlabel('Peso en KG', size=12, family='serif')  
plt.show()

# ### Tamaño de la pantalla
# 

# In[34]:


df["Tamaño de la pantalla"].value_counts().to_frame().head()

# In[35]:


df["Tamaño de la pantalla"] = df["Tamaño de la pantalla"].str.replace('"', "pulgadas")
df["Tamaño de la pantalla"] = df["Tamaño de la pantalla"].str.replace('in', "pulgadas")


# In[36]:


df = separar_valor_um(df = df,
                    colname = "Tamaño de la pantalla", 
                    res_val = 'Valor_Screen', 
                    res_um = 'Medida_Screen',
                    cambio = {"cm" : 2.54,
                              "mm" : 25.4},
                    new_um = "pulgadas")

# In[37]:


df = df[df.Valor_Screen > 8]

# In[38]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,4))

plt.figure(figsize=(10,5))
ax = sns.boxplot(x="Valor_Screen", data=df)
ax.set_title("Tamaño Pantalla", size=14, family='serif') 
plt.xlabel("Tamaño en pulgadas", size=12, family='serif')  
plt.show()

# ### Cuotas

# In[39]:


df.installment.value_counts()

# In[40]:


df.groupby("installment").size().reset_index(name="total")

# In[41]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,4))

plt.figure(figsize=(10,5))
ax = sns.barplot(x="installment", y = "total", data=df.groupby("installment").size().reset_index(name="total"))
ax.set_title("Cuotas sin interés", size=14, family='serif') 
plt.xlabel("Cuotas", size=12, family='serif')  
plt.ylabel("Cantidad PCs", size=12, family='serif')  
plt.show()

# ### Cantidad de nucleos

# In[42]:


df = df[df["Cantidad de núcleos"]<50]

# In[43]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,4))

plt.figure(figsize=(10,5))
ax = sns.boxplot(x="Cantidad de núcleos", data=df[df["Cantidad de núcleos"]<50])
ax.set_title("Cantidad de núcleos", size=14, family='serif') 
plt.xlabel("Cantidad de núcleos procesador", size=12, family='serif')  
plt.show()

# In[44]:


df["Cantidad de núcleos"].describe()

# ### Tarjeta Gráfica

# In[45]:


df["Tarjeta gráfica"].isna().sum()

# In[46]:


df["Tarjeta gráfica"] = df["Tarjeta gráfica"].str.lower()

# In[47]:


lista_dedi = ["nvidia geforce gtx 1650", "gráficos amd radeon™"]

c_nvidia = df["Tarjeta gráfica"].str.contains("nvidia", na = False)
c_dedi = df["Tarjeta gráfica"].isin(lista_dedi)

df["tipo_tarjeta_gráfica"] = "integrada"
df.loc[(c_nvidia | c_dedi), "tipo_tarjeta_gráfica"] = "dedicada"



# In[48]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,5))

ax = sns.boxplot(x="price", data=df, y = "tipo_tarjeta_gráfica")
ax.set_title("Precio por tipo de Tarjeta Gráfica", size=14, family='serif') 
plt.xlabel('Precio)', size=12, family='serif')  
plt.ylabel('Tipo de Tarjeta Gráfica', size=12, family='serif')  
plt.show()

# In[49]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,4))

plt.figure(figsize=(10,5))
ax = sns.barplot(x="tipo_tarjeta_gráfica", y = "total", data=df.groupby("tipo_tarjeta_gráfica").size().reset_index(name="total"))
ax.set_title("Tipo Tarjeta Gráfica", size=14, family='serif') 
plt.xlabel("Tipo", size=12, family='serif')  
plt.ylabel("Cantidad PCs", size=12, family='serif')  
plt.show()

# ### Gamer

# In[50]:


df["Es gamer"].value_counts()

# In[51]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,4))

plt.figure(figsize=(10,5))
ax = sns.barplot(x="Es gamer", y = "total", data=df.groupby("Es gamer").size().reset_index(name="total"))
ax.set_title("Gamer", size=14, family='serif') 
plt.xlabel("Es Gamer", size=12, family='serif')  
plt.ylabel("Cantidad PCs", size=12, family='serif')  
plt.show()

# ### Es 2 en 1
# 

# In[52]:


df["Es 2 en 1"].value_counts()

# In[53]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,4))

plt.figure(figsize=(10,5))
ax = sns.barplot(x="Es 2 en 1", y = "total", data=df.groupby("Es 2 en 1").size().reset_index(name="total"))
ax.set_title("2 en 1", size=14, family='serif') 
plt.xlabel("Es 2 en 1", size=12, family='serif')  
plt.ylabel("Cantidad PCs", size=12, family='serif')  
plt.show()

# ###  Es ultrabook

# In[54]:


df["Es ultrabook"].value_counts()

# In[55]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,4))

plt.figure(figsize=(10,5))
ax = sns.barplot(x="Es ultrabook", y = "total", data=df.groupby("Es ultrabook").size().reset_index(name="total"))
ax.set_title("Ultrabook", size=14, family='serif') 
plt.xlabel("Es ultrabook", size=12, family='serif')  
plt.ylabel("Cantidad PCs", size=12, family='serif')  
plt.show()

# ### Capacidad del disco Sólido

# In[56]:


df = separar_valor_um(df = df,
                    colname = "Capacidad del SSD", 
                    res_val = 'Capacidad_SSD', 
                    res_um = 'Medida_SSD',
                    cambio = {"mb" : 1000,
                              "kb" : 1000000},
                    new_um = "GB")

# In[57]:


df = df[df["Capacidad_SSD"] <= 4000]

# In[58]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,4))

plt.figure(figsize=(10,5))
ax = sns.boxplot(x="Capacidad_SSD", data=df[df["Capacidad_SSD"] <= 4000])
ax.set_title("Capacidad del SSD", size=14, family='serif') 
plt.xlabel("Capacidad del SSD", size=12, family='serif')  
plt.show()

# ### Es touchscreen

# In[59]:


df["Con pantalla táctil"].value_counts()

# In[60]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,4))

plt.figure(figsize=(10,5))
ax = sns.barplot(x="Con pantalla táctil", y = "total", data=df.groupby("Con pantalla táctil").size().reset_index(name="total"))
ax.set_title("Táctil", size=14, family='serif') 
plt.xlabel("Con pantalla táctil", size=12, family='serif')  
plt.ylabel("Cantidad PCs", size=12, family='serif')  
plt.show()

# In[61]:


df.columns

# In[62]:


df = df[["id", "title", "price", "base_price", "initial_quantity", "sold_quantity", "available_quantity", "installment", 
        "shipping_cost", "date_created", "Cantidad de núcleos",  "Tarjeta gráfica", "tipo_tarjeta_gráfica", "Es 2 en 1", "Es gamer",
        "Es ultrabook", "Marca del procesador", "Con pantalla táctil",  "Marca", "linea_procesador", 'Capacidad RAM', 'Valor_Peso', 
        'Valor_Screen', 'tipo_tarjeta_gráfica', 'Capacidad_SSD', 'permalink']]

# ### Manejo de nulos

# In[63]:


df.info(verbose = True)

# In[64]:


df["nulls"] = len(df.columns) - df.apply(lambda x: x.count(), axis=1)

# In[65]:


df.nulls.describe()

# In[66]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,4))

plt.figure(figsize=(10,5))
ax = sns.barplot(x="nulls", y = "total", data=df.groupby("nulls").size().reset_index(name="total"))
ax.set_title("nulls", size=14, family='serif') 
plt.xlabel("nulos", size=12, family='serif')  
plt.ylabel("nulls", size=12, family='serif')  
plt.show()

# ### Filtramos aquellas filas que tengan mas de 2 nulos

# In[67]:


df.shape

# In[68]:


df = df[df["nulls"] <3]

# In[69]:


df.shape

# ### Normalizamos nombres de variables

# In[70]:


import janitor

# In[71]:


df = df.clean_names()

# In[72]:


df = df[df["capacidad_ram"]>0]
df = df[df["valor_peso"]>0]
df = df[df["valor_screen"]>0]	
df = df[df["capacidad_ssd"]>0]	

# ### Imputación de nulos

# In[73]:


df.info()

# Aquellas columnas que son categoricas (true/false) las llenamos asumiendo que lo nulo es falso.

# In[74]:


c_gamer = df.es_gamer.isna()
c_ultra = df.es_ultrabook.isna()
c_21 = df.es_2_en_1.isna()
c_tacil = df.con_pantalla_tactil.isna()

df.loc[c_gamer, "es_gamer"] = "No"
df.loc[c_ultra, "es_ultrabook"] = "No"
df.loc[c_21, "es_2_en_1"] = "No"
df.loc[c_tacil, "con_pantalla_tactil"] = "No"

# Las categoricas las reemplazamos por la moda y las numericas por el promedio.

# In[75]:


import statistics as stat

# In[76]:


c_mp = df.marca_del_procesador.isna()
c_m = df.marca.isna()
c_lp = df.linea_procesador.isna()
c_vp = df.valor_peso.isna()
c_tg = df.tarjeta_grafica.isna()


df.loc[c_mp, "marca_del_procesador"] = stat.mode(df.marca_del_procesador)
df.loc[c_m, "marca"] = stat.mode(df.marca)
df.loc[c_lp, "linea_procesador"] = stat.mode(df.linea_procesador)
df.loc[c_vp, "valor_peso"] = df.valor_peso.median()
df.loc[c_tg, "tarjeta_grafica"] = stat.mode(df.tarjeta_grafica)


# In[77]:


df.info()

# In[78]:


del df["nulls"]

# In[79]:


df._get_numeric_data().head()

# In[80]:



df["ratio_sold"] = df.sold_quantity / (df.sold_quantity + df.available_quantity )

del df["base_price"], df["sold_quantity"], df["initial_quantity"]

# In[81]:


df._get_numeric_data().head()

# In[82]:


df_n = df._get_numeric_data()
columns = df_n.columns

# In[83]:


import datetime as dt

anio = dt.datetime.now().year
mes = dt.datetime.now().month
dia = dt.datetime.now().day
hora = dt.datetime.now().hour
minuto = dt.datetime.now().minute

# In[84]:


df_installment = df.groupby(["id", "installment"]).size().reset_index(name = "q").sort_values(by = "q", ascending = False)
df_installment.head()
df_installment[["id", "installment"]].to_csv(path / f"installments_{anio*10000+mes*100+dia}_{hora}.{minuto}.csv",
index = False, sep = ";")
del df["installment"]

df = df.drop_duplicates()

# In[85]:


df.to_csv(path /f"datos_laptops_transformed_cleaned_{anio*10000+mes*100+dia}_{hora}.{minuto}.csv", 
index = False, sep = ";")

# ## K Means

# In[86]:


from sklearn.cluster import KMeans
import numpy as np

# In[87]:


normalized_df=(df_n-df_n.mean())/df.std()


# In[88]:


normalized_df.head()

# In[89]:


normalized_df.describe()

# In[90]:


normalized_df.info()

# In[91]:


wcss = []
limit = 30
for i in range(1, limit):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=10, random_state=17)
    kmeans.fit(normalized_df)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, limit), wcss)
plt.title('Elbow Method', size=14, family='serif')
plt.xlabel('Number of clusters', size=12, family='serif')  
plt.ylabel('WCSS', size=12, family='serif')  
plt.show()

# In[92]:


kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(normalized_df)

df_n["grupo"] = pred_y

# In[93]:


#sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,4))

plt.figure(figsize=(10,5))
ax = sns.scatterplot(x="capacidad_ram", y = "price", hue = "grupo", data=df_n)
ax.set_title("Kmeans", size=14, family='serif') 
plt.xlabel("RAM", size=12, family='serif')  
plt.ylabel("Precio", size=12, family='serif')  
plt.show()

# In[94]:


df_n["grupo"] = df_n["grupo"].astype(str)

# In[95]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,5))

ax = sns.boxplot(x="price", data=df_n, y = "grupo")
ax.set_title("Precio por grupo", size=14, family='serif') 
plt.xlabel('Precio', size=12, family='serif')  
plt.ylabel('Grupo', size=12, family='serif')  
plt.show()

# In[96]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,5))

ax = sns.boxplot(x="capacidad_ram", data=df_n, y = "grupo")
ax.set_title("RAM", size=14, family='serif') 
plt.xlabel('RAM', size=12, family='serif')  
plt.ylabel('Grupo', size=12, family='serif')  
plt.show()

# In[97]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,5))

ax = sns.boxplot(x="valor_peso", data=df_n, y = "grupo")
ax.set_title("Peso", size=14, family='serif') 
plt.xlabel('Peso', size=12, family='serif')  
plt.ylabel('Grupo', size=12, family='serif')  
plt.show()

# In[98]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,5))

ax = sns.boxplot(x="capacidad_ssd", data=df_n, y = "grupo")
ax.set_title("SSD", size=14, family='serif') 
plt.xlabel('SSD', size=12, family='serif')  
plt.ylabel('Grupo', size=12, family='serif')  
plt.show()

# ## DBSCAN

# In[99]:


from sklearn.neighbors import NearestNeighbors # importing the library
neighb = NearestNeighbors(n_neighbors=2) # creating an object of the NearestNeighbors class
nbrs=neighb.fit(normalized_df) # fitting the data to the object
distances,indices=nbrs.kneighbors(normalized_df)

# In[100]:


# Sort and plot the distances results
distances = np.sort(distances, axis = 0) # sorting the distances
distances = distances[:, 1] # taking the second column of the sorted distances
plt.rcParams['figure.figsize'] = (5,3) # setting the figure size
plt.plot(distances) # plotting the distances
plt.show()

# In[101]:


from sklearn.cluster import DBSCAN


# In[102]:



epsilon = 2
min_samples = 50

# In[103]:


# Compute DBSCAN
db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(normalized_df)
labels = db.labels_

no_clusters = len(np.unique(labels) )
no_noise = np.sum(np.array(labels) == -1, axis=0)

print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of noise points: %d' % no_noise)

# In[104]:


df_n["grupo_dbscan"] = labels

# In[105]:


#sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,4))

plt.figure(figsize=(10,5))
ax = sns.scatterplot(x="capacidad_ssd", y = "price", hue = "grupo_dbscan", data=df_n)
ax.set_title("Kmeans", size=14, family='serif') 
plt.xlabel("capacidad_ssd", size=12, family='serif')  
plt.ylabel("Precio", size=12, family='serif')  
plt.show()

# In[106]:



df_n["grupo_dbscan"] = df_n["grupo_dbscan"].astype(str)


# In[107]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,5))

ax = sns.boxplot(x="price", data=df_n, y = "grupo_dbscan")
ax.set_title("Precio por grupo", size=14, family='serif') 
plt.xlabel('Precio', size=12, family='serif')  
plt.ylabel('Grupo', size=12, family='serif')  
plt.show()

# In[108]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,5))

ax = sns.boxplot(x="capacidad_ram", data=df_n, y = "grupo_dbscan")
ax.set_title("RAM", size=14, family='serif') 
plt.xlabel('RAM', size=12, family='serif')  
plt.ylabel('Grupo', size=12, family='serif')  
plt.show()

# In[109]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,5))

ax = sns.boxplot(x="valor_peso", data=df_n, y = "grupo_dbscan")
ax.set_title("Peso", size=14, family='serif') 
plt.xlabel('Peso', size=12, family='serif')  
plt.ylabel('Grupo', size=12, family='serif')  
plt.show()

# In[110]:


sns.set_theme(style="whitegrid", palette="pastel")

plt.figure(figsize=(8,5))

ax = sns.boxplot(x="capacidad_ssd", data=df_n, y = "grupo_dbscan")
ax.set_title("SSD", size=14, family='serif') 
plt.xlabel('SSD', size=12, family='serif')  
plt.ylabel('Grupo', size=12, family='serif')  
plt.show()

# #### Guardamos un df con las facilidades de pago disponibles, y borramos la columna que indica las cuotas
# Esto lo hacemos para quedarnos solamente con datos pertinentes a la notebook y no inherentes a la publicación.

# In[111]:


df.head()

# ## Gower 

# In[112]:


import gower
df_gower = gower.gower_matrix(df)

# In[113]:


df_gower = pd.DataFrame(df_gower)

# In[114]:


df_gower.head()

# In[115]:


df_gower.shape

# In[116]:


df_gower.to_csv(path / f"gower_distances_{anio*10000+mes*100+dia}_{hora}.{minuto}.csv",
index = False, sep = ";")
