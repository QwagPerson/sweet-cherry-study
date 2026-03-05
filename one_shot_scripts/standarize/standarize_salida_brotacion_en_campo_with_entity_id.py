#############################################################################################################
# Simple script used to standarize "salida_brotacion_en_campo.csv" by mapping text columns to their respective
# maestro IDs. Generates "salida_brotacion_en_campo_with_entity_id.csv" that contains the original data but with an
# additional column "entidad_id" that references the unique entity in the new CSV and removes the original text columns.
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

input_file_path  = data_path / "clean" / "salida_brotacion_en_campo.csv"
output_file_path = "salida_brotacion_en_campo_with_entity_id.csv"

maestro_zonas        = pd.read_csv(data_path / "cleanest" / "maestro_zonas.csv")
maestro_variedades   = pd.read_csv(data_path / "cleanest" / "maestro_variedades.csv")
maestro_tratamientos = pd.read_csv(data_path / "cleanest" / "maestro_tratamientos.csv")
maestro_entidades    = pd.read_csv(data_path / "cleanest" / "maestro_entidades_unicas.csv")

##############################################################################################################
# Main script
##############################################################################################################
# Read the input Excel file
df = pd.read_csv(input_file_path)
df.columns = df.columns.str.lower().str.strip()
df = df.rename(columns={'lugar': 'zona'})

print("Input columns:", df.columns.tolist())
print("Input amount of data:", df.shape)


##############################################################################################################
# Obtener entity_id from maestro_entidades by merging on text columns
##############################################################################################################

df['zona']        = df['zona'].str.lower().str.strip()
df['variedad']    = df['variedad'].str.lower().str.strip()
df['ue']          = df['ue'].astype('str').str.lower().str.strip()

# Fix maestro_entidades ue column type
maestro_entidades['ue'] = maestro_entidades['ue'].astype('str')

# Fix zona spelling inconsistencies
zone_spelling_map = {
    'don sergio'               : 'teno don sergio',
    'teno -sta ana'             : 'santa ana',
    'los niches sta magdalena'  : 'santa magdalena',
    'los niches- marengo'       : 'wapri',
}

df['zona'] = df['zona'].replace(zone_spelling_map)
# Map zona, variedad and tratamiento to their respective maestro ids
df = df.merge(maestro_zonas[['zona', 'zona_id']], on='zona', how='left')
df = df.merge(maestro_variedades[['variedad', 'variedad_id']], on='variedad', how='left')
df = df.merge(maestro_entidades[['entidad_id', 'zona_id', 'variedad_id', 'ue']], on=['zona_id', 'variedad_id', 'ue'], how='left')

##############################################################################################################
# Seleccionamos las columnas finales y guardamos el resultado
##############################################################################################################

df = df [[
    'entidad_id',
    'fecha', 'momento',
    'n° dardo', 'n° yemas', 'n° brot'
]]

df = df.rename(columns={
    'n° dardo' : 'numero_dardo',
    'n° yemas' : 'cantidad_yemas',
    'n° brot'  : 'cantidad_yemas_brotadas',
})

print("Output columns:", df.columns.tolist())
print("Output amount of data:", df.shape)
df.to_csv(output_file_path, index=False)
