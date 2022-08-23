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