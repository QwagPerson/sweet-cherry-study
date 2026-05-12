#############################################################################################################
# Simple script that uses entrada_caida_de_hojas_with_entity_id.csv to generate a new CSV that contains the latency data
# for each unique entity found in entrada_caida_de_hojas.csv. The latency data is the first found date where the
# accumulated percentage of leaf fall (d%) is greater than or equal to 50%.
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

input_file_path  = "entrada_caida_de_hojas_with_entity_id.csv"
output_file_path = "entrada_latencia_with_entity_id.csv"

maestro_zonas        = pd.read_csv(data_path / "cleanest" / "maestro_zonas.csv")
maestro_variedades   = pd.read_csv(data_path / "cleanest" / "maestro_variedades.csv")
maestro_tratamientos = pd.read_csv(data_path / "cleanest" / "maestro_tratamientos.csv")

##############################################################################################################
# Main script
##############################################################################################################

df = pd.read_csv(input_file_path)
ue_to_latency = []

for entidad_id, group in df.groupby('entidad_id'):
    # Find the first date where porcentaje_hojas_caidas >= 50%
    latency_row = group[group['porcentaje_hojas_caidas'] >= 50.0].sort_values(by='fecha').head(1)
    if not latency_row.empty:
        latency_date = latency_row['fecha'].values[0]
        ue_to_latency.append({'entidad_id': entidad_id, 'fecha_latencia': latency_date})

latency_df = pd.DataFrame(ue_to_latency)
latency_df.to_csv(output_file_path, index=False)


