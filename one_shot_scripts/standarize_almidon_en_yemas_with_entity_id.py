#############################################################################################################
# Simple script used to standarize "almidon_en_yemas.csv" by mapping text columns to their respective
# maestro IDs. Generates "almidon_en_yemas_with_entity_id.csv" that contains the original data but with an
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

input_file_path  = data_path / "raw" / "FIA Entrada 2024 - Almidón en yemas.xlsx"
output_file_path = "entrada_almidon_en_yemas.csv"

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
df = pd.read_excel(input_file_path)
df.columns = df.columns.str.lower().str.strip()

# The columns that define a unique entity have a slightly different naming here so we must first rename them
df.rename(columns={'lugar': 'zona'}, inplace=True)

# Also their values have slighty different spelling so here we standarize them
df['zona']       = df['zona'].str.lower().str.strip()
df['variedad']    = df['variedad'].str.lower().str.strip()
df['tratamiento'] = df['tratamiento'].str.lower().str.strip()
df['ue']          = df['ue'].astype('str').str.lower().str.strip()

zone_spelling_map = {
    'teno -prado'               : 'teno prado',
    'teno -sta ana'             : 'santa ana',
    'los niches sta magdalena'  : 'santa magdalena',
    'los niches- marengo'       : 'wapri',
}

df['zona'] = df['zona'].replace(zone_spelling_map)

# Map zona, variedad and tratamiento to their respective maestro ids
df = df.merge(maestro_zonas[['zona', 'zona_id']], on='zona', how='left')
df = df.merge(maestro_variedades[['variedad', 'variedad_id']], on='variedad', how='left')
df = df.merge(maestro_tratamientos[['tratamiento', 'tratamiento_id']], on='tratamiento', how='left')
df = df.merge(maestro_entidades[['entidad_id', 'zona_id', 'variedad_id', 'tratamiento_id', 'ue']],
              on=['zona_id', 'variedad_id', 'tratamiento_id', 'ue'], how='left')

# Select only the columns we are interested
df = df[['entidad_id', 'fecha', 'muestreo', 'peso(mg)', 'absorbancia', 'conc. mg/g']]

# Rename columns to more descriptive names
df.rename(
    columns={
        'peso(mg)'  : 'peso_yema_mg',
        'absorbancia': 'absorbancia',
        'conc. mg/g' : 'concentracion_almidon_mg_por_g'
    },
    inplace=True
)

# Fill with nan not numeric values over those 3 columns (peso, absorbancia, concentracion)
print(df.columns)
df['peso_yema_mg'] = pd.to_numeric(df['peso_yema_mg'], errors='coerce')
df['absorbancia']  = pd.to_numeric(df['absorbancia'], errors='coerce')
df['concentracion_almidon_mg_por_g'] = pd.to_numeric(df['concentracion_almidon_mg_por_g'], errors='coerce')

df.to_csv(output_file_path, index=False)

print(df.head())
