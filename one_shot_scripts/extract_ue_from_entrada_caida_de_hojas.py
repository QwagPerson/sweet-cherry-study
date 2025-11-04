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

input_file_path  = "../entrada_caida_de_hojas.csv"
output_file_path = "maestro_entidades_unicas.csv"

maestro_zonas        = pd.read_csv(data_path / "cleanest" / "maestro_zonas.csv")
maestro_variedades   = pd.read_csv(data_path / "cleanest" / "maestro_variedades.csv")
maestro_tratamientos = pd.read_csv(data_path / "cleanest" / "maestro_tratamientos.csv")

##############################################################################################################
# Main script
##############################################################################################################

df = pd.read_csv(input_file_path)

# This 4 columns are the ones that define a unique entity
df['lugar']       = df['lugar'].str.lower().str.strip()
df['variedad']    = df['variedad'].str.lower().str.strip()
df['tratamiento'] = df['tratamiento'].str.lower().str.strip()
df['ue']          = df['ue'].astype('str').str.lower().str.strip()

df.rename(columns={'lugar': 'zona'}, inplace=True)

entities = df[['zona', 'variedad', 'tratamiento', 'ue']].drop_duplicates()

# Map zona, variedad and tratamiento to their respective maestro ids
entities = entities.merge(maestro_zonas[['zona', 'zona_id']], on='zona', how='left')
entities = entities.merge(maestro_variedades[['variedad', 'variedad_id']], on='variedad', how='left')
entities = entities.merge(maestro_tratamientos[['tratamiento', 'tratamiento_id']], on='tratamiento', how='left')

# Lets use the index as unique id for each entity
entities = entities.reset_index(drop=True)
entities['entidad_id'] = entities.index

# Now lets generate the entrada_caida_de_hojas_with_entity_id.csv
df = df.merge(entities, on=['zona', 'variedad', 'tratamiento', 'ue'], how='left')
df = df.drop(columns=['zona', 'variedad', 'tratamiento', 'ue'])

# Select only the columns we are interested
df = df[['entidad_id', 'fecha', 'a %', 'va %', 'v%', 'ch %', 'd%']]
df.rename(
    columns={
        'a %'  : 'porcentaje_hojas_amarillas',
        'va %' : 'porcentaje_hojas_verde_amarillas',
        'v%'   : 'porcentaje_hojas_verdes',
        'ch %' : 'porcentaje_hojas_caidas',
        'd%'   : 'porcentaje_hojas_danadas'
    },
    inplace=True
)

df.to_csv("entrada_caida_de_hojas_with_entity_id.csv", index=False)

# Lets reorder the columns and drop the original text columns
entities = entities[['entidad_id', 'zona_id', 'variedad_id', 'tratamiento_id', 'ue']]
entities.rename({'ue': 'unidad_experimental'}, inplace=True)
# Now lets generate the output CSV
entities.to_csv(output_file_path, index=False)

