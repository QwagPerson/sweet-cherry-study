"""
FIA 2024 - T50 Brotación - Normalization and Cross-join Script

Este módulo procesa datos de hojas de excel del archivo 'FIA 2024 - T50 Brotación.xlsx'
y genera dos archivos CSV consolidados:
  - df_frio: Cruce entre datos de entrada (Caída de hoja ENT + Brotación en cámara ENT)
  - df_calor: Cruce entre datos de salida (Brotación en campo SLD + Brotación en cámara SLD)

El script normaliza valores de tratamiento, variedad y lugar, filtra solo controles,
y genera una clave única (key) para realizar merges entre dataframes.

Archivos utilizados:
    - ORIGIN_FILE_PATH: data/0_bronze/FIA 2024 - T50 Brotación.xlsx
    - OUTPUT_PATH: data/1_silver/normalize_temps_2024

Autor: Data Science Team
"""

import pathlib as pl
import pandas as pd

def normalize_tratamiento(df):
    tratamiento_map = {
        'ABA'                                   : 'ABA',
        'Control'                               : 'Control',
        'Ethephon'                              : 'ETH',
        'Fsh'                                   : 'FSH',
        'Zn+Eth'                                : 'ETH + ZN',
        'Zn+U+Eth'                              : 'ETH + ZN + U',
        # Confirmar dosis? de esto
        'DORMEX 2% + 2% Ac. Mineral'            : 'C.H. 1% (i.a.) + Ac. Min. 2%',
        'DORMEX 3% + BREAK'                     : 'C.H. 1% (i.a.)',
        'ERGER 5% + Nca 6%'                     : 'Erger 5% (p.c.) + N. Ca 6%',
        'CH+Break'                              : 'C.H. 1% (i.a.)',
        'CH+AcMin'                              : 'C.H. 1% (i.a.) + Ac. Min. 2%',
        'ERG+NCa'                               : 'Erger 5% (p.c.) + N. Ca 6%',
    }

    if 'Tratamiento' not in df.columns:
        raise ValueError(f"Expected Tratamiento column in dataframe, but not found. Columns: {df.columns}")

    for x in df['Tratamiento'].unique():
        if x not in tratamiento_map.keys():
            raise ValueError(f"Valor '{x}' no tiene mapeo en tratamiento_map")

    df['Tratamiento'] = df['Tratamiento'].map(tratamiento_map)

    return df

def normalize_variedad(df):
    variedad_map = {
        'Santina' : 'santina',
        'Kordia'  : 'kordia',
        'Regina'  : 'regina',
    }

    if 'Variedad' not in df.columns:
        raise ValueError(f"Expected Variedad column in dataframe, but not found. Columns: {df.columns}")

    for x in df['Variedad'].unique():
        if x not in variedad_map.keys():
            raise ValueError(f"Valor {x} no tiene mapeo en variedad_map")

    df['Variedad'] = df['Variedad'].map(variedad_map)

    return df

def normalize_lugar(df):
    lugar_map = {
     'Melipilla'                     : 'melipilla',
    'Pichidegua'                    : 'pichidegua',
    'Viluco'                        : 'viluco',
    'Teno Poniente'                 : 'teno poniente',
    'Teno Oriente'                  : 'teno oriente',
    'Sarmiento'                     : 'sarmiento',
    'Teno Montaña'                  : 'teno montaña',
    'Wapri - Los Niches'            : 'los niches',
    'Sta Magdalena - Los Niches'    : 'los niches',
    'Teno poniente'                 : 'teno poniente',
    'Los Niches'                    : 'los niches',
    'Los Niches 2'                  : 'los niches',
    'Los Niches 3'                  : 'los niches',
    'Los Niches 4'                  : 'los niches',
    'Los Niches 5'                  : 'los niches',
    'Los Niches 6'                  : 'los niches',
    'Los Niches 7'                  : 'los niches',
    'Los Niches 8'                  : 'los niches',
    'Los Niches 9'                  : 'los niches',
    'Los Niches 10'                 : 'los niches',
    'Los Niches 11'                 : 'los niches',
    'Los Niches 12'                 : 'los niches',
    'Los Niches 13'                 : 'los niches',
    'Los Niches 14'                 : 'los niches',
    'Los Niches 15'                 : 'los niches',
    'Los Niches 16'                 : 'los niches',
    'Los Niches 17'                 : 'los niches',
    'Los Niches 18'                 : 'los niches',
    'Los Niches 19'                 : 'los niches',
    'Los Niches 20'                 : 'los niches',
    'Los Niches 21'                 : 'los niches',
    }

    if 'Lugar' not in df.columns:
        raise ValueError(f"Expected Lugar column in dataframe, but not found. Columns: {df.columns}")

    for x in df['Lugar'].unique():
        if x not in lugar_map.keys():
            raise ValueError(f"Valor {x} no tiene mapeo en lugar_map")

    df['Lugar'] = df['Lugar'].map(lugar_map)

    return df


ORIGIN_FILE_PATH = pl.Path(__file__).parent / '..' / 'data' / '0_bronze' / 'FIA 2024 - T50 Brotación.xlsx'
OUTPUT_PATH      = pl.Path(__file__).parent / '..' / 'data' / '1_silver' / 'normalize_temps_2024'

df_caida_hoja_ent = pd.read_excel(
    ORIGIN_FILE_PATH, sheet_name='Caída de hoja ENT'
)

df_brotacion_camara_ent = pd.read_excel(
    ORIGIN_FILE_PATH, sheet_name='Brotación en cámara ENT'
)

df_brotacion_campo_salida = pd.read_excel(
    ORIGIN_FILE_PATH, sheet_name='Brotación en campo SLD'
)

df_brotacion_camara_salida = pd.read_excel(
    ORIGIN_FILE_PATH, sheet_name='Brotación en cámara SLD'
)

dfs = [
    df_caida_hoja_ent,
    df_brotacion_camara_ent,
    df_brotacion_campo_salida,
    df_brotacion_camara_salida,
]

for i, df in enumerate(dfs):
    df        = normalize_lugar(df)
    df        = normalize_variedad(df)
    df        = normalize_tratamiento(df)
    df['UE']  = df['UE'].astype(int).astype(str)
    df['key'] = df['Lugar'] + '_' + df['Variedad'] + '_' + df['Tratamiento'] + '_' + df['UE']
    df        = df.loc[df['Tratamiento'] == 'Control']
    dfs[i]    = df


# FRIO

df_frio = dfs[0].merge(
    dfs[1],
    on='key',
    suffixes=('_caida_hoja', '_brotacion_camara')
)
print(df_frio.columns)
df_frio.to_csv("test.csv", index=False)


# CALOR

df_calor = dfs[2].merge(
    dfs[3],
    on='key',
    suffixes=('_brotacion_campo', '_brotacion_camara_salida')
)
