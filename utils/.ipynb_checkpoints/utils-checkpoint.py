import pandas as pd
def get_attributes_from_row(row):
  fila = dict()
  for e in row:
    for element in e:
      #print(element)
      fila[str(element['name'])] =  element['value_name']
    
  return pd.DataFrame.from_dict(fila, orient='index').transpose()