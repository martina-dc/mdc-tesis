from ast import If
from typing import List
from typing_extensions import Self
import pandas as pd
import numpy as np
import json
import requests

from datetime import datetime, timedelta, timezone, date
import os
from pyprojroot import here

path_secrets = here() / "secrets"     
meli_auth_token = None

def get_attributes_from_row(row):
    fila = dict()
    for e in row:
        for element in e:
            #print(element)
            fila[str(element['name'])] =  element['value_name']
    
    return pd.DataFrame.from_dict(fila, orient='index').transpose()


def auth_token():
    with open(path_secrets / 'secrets.json') as f:
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
    
def get_available_filters()-> dict:

    """
    Esta funcion devuelve los filtros disponibles para las publicaciones de notebooks que son buscadas en la pagina de mercado libre bajo algun criterio.
    No parameters.
    Returns:
    
        dict: Diccionario que contiene los filtros disponibles para la categoria.
    """
    url='https://api.mercadolibre.com/sites/MLA/search?category=MLA1652&search?include_filters=true#json'
    r = get_response(url) 
    return r.json()["available_filters"] #Guardamos el número máximo de resultados para luego stoppear nuestro loop


def get_q_items(filters: dict = {}) -> int:
    """
    Esta funcion devuelve la cantidad de publicaciones de notebooks que son buscadas en la pagina de mercado libre bajo algun criterio.
    Se pueden enviar como maximo 3 filtros. Si se envian mas de 3 solo se tomaran los 3 primeros. 
    Parameters:
        filters: dict. Este diccionario tiene que tener la clave del tipo de filtro que se quiere usar y el valor del filtro seleccionado.
        Por ejemplo:
            filters: {"installments" : "yes"} seria publicaciones con cuotas sin interes.
    
    """
    if len(filters) == 0:
        url='https://api.mercadolibre.com/sites/MLA/search?category=MLA1652&search_type=scan&ITEM_CONDITION=2230284#json'
    elif len(filters) == 1:
        iter_f = iter(filters)
        url=f'https://api.mercadolibre.com/sites/MLA/search?category=MLA1652&search_type=scan&ITEM_CONDITION=2230284&{next(iter_f)}={filters[next(iter_f)]}#json'
    elif len(filters) == 2:
        iter_f = iter(filters)
        id1 = next(iter_f)
        id2 = next(iter_f)
        url=   'https://api.mercadolibre.com/sites/MLA/search?category=MLA1652&search_type=scan&ITEM_CONDITION=2230284' + \
                f'&{id1}={filters[next(id1)]}'+ \
                f'&{id2}={filters[next(id2)]}'+ \
                '&order_by=start_time_desc&#json'
    elif len(filters) == 3:
        iter_f = iter(filters)
        id1 = next(iter_f)
        id2 = next(iter_f)
        id3 = next(iter_f)
        url=   'https://api.mercadolibre.com/sites/MLA/search?category=MLA1652&search_type=scan&ITEM_CONDITION=2230284' + \
                f'&{id1}={filters[id1]}'+ \
                f'&{id2}={filters[id2]}'+ \
                f'&{id3}={filters[id3]}'+ \
                '&order_by=start_time_desc&#json'       
   
    r = get_response(url) 
    maximum = int(str(r.json()["paging"]["total"])) #Guardamos el número máximo de resultados para luego stoppear nuestro loop
    return maximum

    
def create_item_list(filters: dict = {}):
    """
    Esta funcion devuelve una lista de publicaciones de notebooks que son buscadas en la pagina de mercado libre bajo algun criterio.
    Se pueden enviar como maximo 3 filtros. Si se envian mas de 3 solo se tomaran los 3 primeros. 
    Parameters:
        filters: dict. Este diccionario tiene que tener la clave del tipo de filtro que se quiere usar y el valor del filtro seleccionado.
        Por ejemplo:
            filters: {"installments" : "yes"} seria publicaciones con cuotas sin interes.
    
    """
    
    item_list = []
    maximum = get_q_items(filters)
    for offset in range(0,maximum,50):
        if offset >= 4000:
            print(f"El offst es {offset} se terminará la ejecución.")
            break
        
        
        if len(filters) == 0:
               url = f'https://api.mercadolibre.com/sites/MLA/search?category=MLA1652&offset={offset}&ITEM_CONDITION=2230284&order_by=start_time_desc&#json'
        elif len(filters) == 1:
            iter_f = iter(filters)
            url = f'https://api.mercadolibre.com/sites/MLA/search?category=MLA1652&offset={offset}&ITEM_CONDITION=2230284&order_by=start_time_desc&{next(iter_f)}={filters[next(iter_f)]}#json'

        elif len(filters) == 2:
            iter_f = iter(filters)
            id1 = next(iter_f)
            id2 = next(iter_f)

            url = f'https://api.mercadolibre.com/sites/MLA/search?category=MLA1652&offset={offset}&ITEM_CONDITION=2230284&order_by=start_time_desc&' + \
                    f'&{id1}={filters[id2]}'+ \
                    f'&{id2}={filters[id2]}'+ \
                    '#json'
        elif len(filters) == 3:
            iter_f = iter(filters)
            id1 = next(iter_f)
            id2 = next(iter_f)
            id3 = next(iter_f)
            url= f'https://api.mercadolibre.com/sites/MLA/search?category=MLA1652&offset={offset}&ITEM_CONDITION=2230284&order_by=start_time_desc&' + \
                    f'&{id1}={filters[id1]}'+ \
                    f'&{id2}={filters[id2]}'+ \
                    f'&{id3}={filters[id3]}'+ \
                    '#json'
        r = get_response(url)
        if r is not None and r.status_code == 200:
            
            if r is not None:
                data = r.json()
                length = len(data['results'])
                for i in range(length):
                    item_id = data['results'][i]['id']
                    item_list.append(item_id)
        
        print(f"Porcentaje de completitud: {(offset/maximum):0.2%}",end='\r')

            

    return item_list


def get_df_list(item_list: list)-> pd.DataFrame():
    """
    Esta funcion nos permite obtener información detallada de cada item que vamos a descargar.
    Parameters:
        -item_list: Es una lista de items que se deben buscar.
        
    Return: pd.DataFrame con los archivos encontrados.
    
    """
    final_list = []
    for i in range(len(item_list)):
        item=f"https://api.mercadolibre.com/items/{item_list[i]}"
        item_add = requests.get(item)
        item_add = item_add.json()
        final_list.append(item_add)
        print(f"Porcentaje de completitud: {(i+1)/len(item_list):0.2%}",end='\r')
    return pd.json_normalize(final_list)


def normalizar_lineas_procesador(df: pd.DataFrame, col_original: str)-> pd.DataFrame:
    """
    Esta funcion recibe un dataframe y devuelve el mismo DF con una columna de linea de procesador que esta normalizada.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe que debe tener una colummna llamada linea_procesador_

    Returns
    -------
    pd.DataFrame
        Dataframe con columna cambiada
    """
    condic_i3 = df[col_original].str.lower().str.contains("i3", na=False)
    condic_i5 = df[col_original].str.lower().str.contains("i5", na=False)
    condic_i7 = df[col_original].str.lower().str.contains("i7", na=False)
    condic_i9 = df[col_original].str.lower().str.contains("i9", na=False)

    df["linea_procesador_"] = df[col_original].copy()
    df.loc[condic_i3, "linea_procesador_"] = "Core i3"
    df.loc[condic_i5, "linea_procesador_"] = "Core i5"
    df.loc[condic_i7, "linea_procesador_"] = "Core i7"
    df.loc[condic_i9, "linea_procesador_"] = "Core i9"

    cond_ryzen = df[col_original].str.lower().str.contains("ryzen", na=False)
    condic_r3 = df[col_original].str.lower().str.contains("3", na=False)
    condic_r5 = df[col_original].str.lower().str.contains("5", na=False)
    condic_r7 = df[col_original].str.lower().str.contains("7", na=False)
    condic_r9 = df[col_original].str.lower().str.contains("9", na=False)


    df.loc[condic_r3 & cond_ryzen, "linea_procesador_"] = "Ryzen 3"
    df.loc[condic_r5& cond_ryzen, "linea_procesador_"] = "Ryzen 5"
    df.loc[condic_r7& cond_ryzen, "linea_procesador_"] = "Ryzen 7"
    df.loc[condic_r9& cond_ryzen, "linea_procesador_"] = "Ryzen 9"

    condic_a6 = df[col_original].str.lower().str.contains("a6", na=False)
    condic_a8 = df[col_original].str.lower().str.contains("a8", na=False)
    condic_a10 = df[col_original].str.lower().str.contains("a10", na=False)
    condic_a12 = df[col_original].str.lower().str.contains("a12", na=False)


    df.loc[condic_a6, "linea_procesador_"] = "AMD A6"
    df.loc[condic_a8, "linea_procesador_"] = "AMD A8"
    df.loc[condic_a10, "linea_procesador_"] = "AMD A10"
    df.loc[condic_a12, "linea_procesador_"] = "AMD A12"

    cond_celeron = df[col_original].str.lower().str.contains("celeron", na=False)
    df.loc[cond_celeron, "linea_procesador_"] = "Celeron"

    cond_pentium = df[col_original].str.lower().str.contains("pentium", na=False)
    df.loc[cond_pentium, "linea_procesador_"] = "Pentium"

    cond_athlon = df[col_original].str.lower().str.contains("athlon", na=False)
    df.loc[cond_athlon, "linea_procesador_"] = "Athlon"

    cond_sempron = df[col_original].str.lower().str.contains("sempron", na=False)
    df.loc[cond_sempron, "linea_procesador_"] = "Sempron"

    cond_m1 = df[col_original].str.lower().str.contains("m1", na=False)
    df.loc[cond_m1, "linea_procesador_"] = "M1"

    cond_m2 = df[col_original].str.lower().str.contains("m2", na=False)
    df.loc[cond_m2, "linea_procesador_"] = "M2"

    proc = df.linea_procesador.value_counts().to_frame()
    proc = df.linea_procesador_.value_counts().to_frame()
    proc["condicion"] = proc.index
    proc.loc[proc["linea_procesador_"] < 40, "condicion"] = "Otro"

    proc = proc.rename_axis('linea_procesador_Original').reset_index()
    proc.columns = ["linea_procesador_Original", "Cantidad", "linea_procesador_Nueva"]
    del proc["Cantidad"]

    df = df.merge(proc,
        how = "left",
        left_on = "linea_procesador_",
        right_on = "linea_procesador_Original")

    del df["linea_procesador_Original"],  df[col_original], df["linea_procesador_"]


    df = df.rename(columns = {"linea_procesador_Nueva"  : col_original}) 
    

    return df



def separar_valor_ram(df: pd.DataFrame, colname: str, res_val: str, res_um: str)->pd.DataFrame:
    """
    Esta funcion recibe una columna que tiene un valor y una unidad de medida y lo devuelve en dos columnas
    separadas. El nombre de la columna a tomar es colname y lo devuelve en res_val el numero y en res_um la unidad de medida. 
  

    Parameters
    ----------
    df : pd.DataFrame
        df con la columna a desglosar.
    colname : str
        nombre de la columna del df que tiene el valor combinado.
    res_val : str
        nuevo nombre que se le desea poner a la columna que tiene el valor.
    res_um : str
       nuevo nombre que se le desea poner a la columna que tiene la unidad de medida.

    Returns
    -------
    pd.DataFrame
        _description_
    """

    lista = df[colname].str.split(' ', 1, expand=True)
    df[res_val], df[res_um] = lista[0], lista[1]
    cond_ram_mb = (df[res_um].str.lower() == "mb")
    df.loc[cond_ram_mb, res_val] = df[res_val].fillna(-1).astype(float, errors = "ignore").astype(int) / 1000
    df.loc[cond_ram_mb, res_um] = "GB"

    cond_ram_kb = (df[res_um].str.lower() == "kb")
    df.loc[cond_ram_kb, res_val] = df[res_val].fillna(-1).astype(float, errors = "ignore").astype(int) / 1000000
    df.loc[cond_ram_kb, res_um] = "GB"

    df[res_val] = df[res_val].fillna(-1).astype(float)

    return df