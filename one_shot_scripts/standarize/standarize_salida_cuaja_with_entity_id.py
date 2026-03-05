#############################################################################################################
# Simple script used to standarize "salida_cuaja.csv" by mapping text columns to their respective
# maestro IDs. Generates "salida_cuaja.csv" that contains the original data but with an
# additional column "entity_id" that references the unique entity in the new CSV and removes the original text columns.
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

input_file_path  = data_path / "clean" / "salida_cuaja.csv"
output_file_path = "salida_cuaja_with_entity_id.csv"

maestro_zonas        = pd.read_csv(data_path / "cleanest" / "maestro_zonas.csv")
maestro_variedades   = pd.read_csv(data_path / "cleanest" / "maestro_variedades.csv")
maestro_tratamientos = pd.read_csv(data_path / "cleanest" / "maestro_tratamientos.csv")
maestro_entidades    = pd.read_csv(data_path / "cleanest" / "maestro_entidades_unicas.csv")

##############################################################################################################
# Main script
##############################################################################################################
# Fix maestro_entidades ue column type
maestro_entidades['ue'] = maestro_entidades['ue'].astype('str')

# Read the input Excel file
df = pd.read_csv(input_file_path)
df.columns = df.columns.str.lower().str.strip()

print("Input amount of data:", df.shape)

# The columns that define a unique entity have a slightly different naming here so we must first rename them
df.rename(columns={'lugar': 'zona'}, inplace=True)

# Also their values have slighty different spelling so here we standarize them
df['zona']       = df['zona'].str.lower().str.strip()
df['variedad']    = df['variedad'].str.lower().str.strip()
df['ue']          = df['ue'].astype('str').str.lower().str.strip()

zone_spelling_map = {
    'teno -prado'                  : 'teno prado',
    'teno -sta ana'                : 'santa ana',
    'los niches sta magdalena'     : 'santa magdalena',
    'los niches- marengo'          : 'wapri',
    'teno santa ana'               : 'santa ana',
    'los niches wapri'             :  'wapri',
    'los niches santa magdalena'   : 'santa magdalena'
}

df['zona'] = df['zona'].replace(zone_spelling_map)

# Map zona, variedad and tratamiento to their respective maestro ids
df = df.merge(maestro_zonas[['zona', 'zona_id']], on='zona', how='left')
df = df.merge(maestro_variedades[['variedad', 'variedad_id']], on='variedad', how='left')
df = df.merge(maestro_entidades[['entidad_id', 'zona_id', 'variedad_id', 'ue']],
              on=['zona_id', 'variedad_id', 'ue'], how='left')

print(df.columns)
# Select only the columns we are interested
df = df[[
    'entidad_id',
    'momento', 'fecha flores', 'fecha frutos',
    'n° dardo', 'nº flores', 'nº frutos', '%cuaja'
]]

# Rename columns to more descriptive names
df.rename(
    columns={
        'fecha flores'         : 'fecha_flores',
        'fecha frutos'         : 'fecha_frutos',
        'n° dardo'             : 'numero_dardos',
        'nº flores'            : 'numero_flores',
        'nº frutos'            : 'numero_frutos',
        '%cuaja'               : 'porcentaje_cuaja'
    },
    inplace=True
)

# Replace '-' with nan
df.replace('-', pd.NA, inplace=True)


# Fill with nan with 0 as we assume missing data means no flowers or fruits
df['numero_flores'] = df['numero_flores'].fillna(0)
df['numero_frutos'] = df['numero_frutos'].fillna(0)
df['porcentaje_cuaja'] = df['porcentaje_cuaja'].fillna(0)



print("Output amount of data:", df.shape)
print(df.columns)
print(df.head())

df.to_csv(output_file_path, index=False)



