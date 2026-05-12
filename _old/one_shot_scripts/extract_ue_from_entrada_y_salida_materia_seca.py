#############################################################################################################
# Simple script used to extract unique entities from "entrada_caida_de_hojas.csv" and to generate a new CSV
# that contains only the unique entities. Also generates entrada_caida_de_hojas_with_entity_id.csv that
# contains the original data  but with an additional column "entity_id" that references the unique entity
# in the new CSV and removes the original text columns.
##############################################################################################################

##############################################################################################################
# Imports
##############################################################################################################

import pandas as pd
import pathlib as pl

##############################################################################################################
# Constants
##############################################################################################################
data_path = pl.Path("D:/Data/frusan")

input_entrada_file_path  = data_path / "clean" / "entrada_materia_seca.csv"
input_salida_file_path  = data_path / "clean" / "salida_materia_seca.csv"

output_maestro_entidades_file_path = "maestro_entidades_unicas.csv"

maestro_zonas        = pd.read_csv(data_path / "cleanest" / "maestro_zonas.csv")
maestro_variedades   = pd.read_csv(data_path / "cleanest" / "maestro_variedades.csv")
maestro_tratamientos = pd.read_csv(data_path / "cleanest" / "maestro_tratamientos.csv")

maestro_tratamientos_entrada = maestro_tratamientos.loc[maestro_tratamientos.tipo=='entrada']
maestro_tratamientos_salida  = maestro_tratamientos.loc[maestro_tratamientos.tipo=='salida']


##############################################################################################################
# Main script
##############################################################################################################

df_entrada = pd.read_csv(input_entrada_file_path)
df_entrada.columns = df_entrada.columns.str.lower().str.replace(" ", "")

df_salida  = pd.read_csv(input_salida_file_path)
df_salida.columns = df_salida.columns.str.lower().str.replace(" ", "")

print(f"Entrada shape: {df_entrada.shape}")
print(f"Salida shape: {df_salida.shape}")
print(f"Columns in entrada: {df_entrada.columns.tolist()}")
print(f"Columns in salida: {df_salida.columns.tolist()}")

# This 3 columns are the ones that define a unique entity
df_entrada['lugar']       = df_entrada['lugar'].str.lower().str.strip()
df_entrada['variedad']    = df_entrada['variedad'].str.lower().str.strip()
df_entrada['ue']          = df_entrada['ue'].astype('str').str.lower().str.strip()
df_entrada['llave']       = df_entrada['lugar'] + '|' + df_entrada['variedad'] + '|' + df_entrada['ue']
df_entrada.rename(columns={'lugar': 'zona'}, inplace=True)

df_entrada['tratamiento'] = df_entrada['tratamiento'].str.lower().str.strip()


entities_entrada = df_entrada[['llave']].drop_duplicates()
print("Unique entities found in entrada:", entities_entrada['llave'].nunique())

# Now the same for salida
df_salida['lugar']       = df_salida['lugar'].str.lower().str.strip()
df_salida['variedad']    = df_salida['variedad'].str.lower().str.strip()
df_salida['ue']          = df_salida['ue'].astype('str').str.lower().str.strip()
df_salida['llave']       = df_salida['lugar'] + '|' + df_salida['variedad'] + '|' + df_salida['ue']
df_salida.rename(columns={'lugar': 'zona'}, inplace=True)

df_salida['tratamiento'] = df_salida['tratamiento'].str.lower().str.strip()

entities_salida = df_salida[['llave']].drop_duplicates()

print("Unique entities found in salida:", entities_salida['llave'].nunique())

print("Entities in entrada not in salida:", entities_entrada[~entities_entrada['llave'].isin(entities_salida['llave'])].head())
print("Entities in salida not in entrada:", entities_salida[~entities_salida['llave'].isin(entities_entrada['llave'])].head())

entities = pd.concat([entities_entrada, entities_salida], ignore_index=True).drop_duplicates()
print("Total unique entities (entrada + salida):", entities.shape[0])

# Now we need to split the llave back to its components
entities[['zona', 'variedad', 'ue']] = entities['llave'].str.split('|', expand=True)
entities = entities.drop(columns=['llave'])

# Recover the tratamiento information from entrada and salida
entities = entities.merge(
    df_entrada[['zona', 'variedad', 'ue', 'tratamiento']].drop_duplicates(),
    on=['zona', 'variedad', 'ue'],
    how='left'
)
entities = entities.merge(
    df_salida[['zona', 'variedad', 'ue', 'tratamiento']].drop_duplicates(),
    on=['zona', 'variedad', 'ue'],
    how='left',
    suffixes=('_entrada', '_salida')
)

# If tratamiento is missing in salida, mark as 'N/A'
entities['tratamiento_salida'] = entities['tratamiento_salida']
entities['tratamiento_entrada'] = entities['tratamiento_entrada']

# Map zona, variedad and tratamiento to their respective maestro ids
entities = entities.merge(maestro_zonas[['zona', 'zona_id']], on='zona', how='left')
entities = entities.merge(maestro_variedades[['variedad', 'variedad_id']], on='variedad', how='left')

# Use a map for tratamiento
tratamiento_entrada_map = {}
for _, row in maestro_tratamientos_entrada.iterrows():
    tratamiento_entrada_map[row['tratamiento']] = row['tratamiento_id']

tratamiento_salida_map = {}
for _, row in maestro_tratamientos_salida.iterrows():
    tratamiento_salida_map[row['tratamiento']] = row['tratamiento_id']

entities['tratamiento_entrada_id'] = entities['tratamiento_entrada'].apply(lambda x : tratamiento_entrada_map[x])
entities['tratamiento_salida_id']  = entities['tratamiento_salida'].apply(lambda x : tratamiento_salida_map[x])

# Lets use the index as unique id for each entity
entities = entities.reset_index(drop=True)
entities['entidad_id'] = entities.index


# Lets reorder the columns and drop the original text columns
entities = entities[[
    'entidad_id',
    'zona_id', 'zona',
    'variedad_id', 'variedad',
    'tratamiento_entrada_id', 'tratamiento_entrada',
    'tratamiento_salida_id', 'tratamiento_salida',
    'ue']]
entities.rename({'ue': 'unidad_experimental'}, inplace=True)

# Now lets generate the output CSV
entities.to_csv(output_maestro_entidades_file_path, index=False)