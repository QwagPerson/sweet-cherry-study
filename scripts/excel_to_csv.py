import pandas as pd
import pathlib as pl

def excel_to_csv(input_excel_path: pl.Path, output_csv_path: pl.Path):
    xl = pd.ExcelFile(input_excel_path)
    df_list = []
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        df_list.append(df)

    concatenated_df = pd.concat(df_list, ignore_index=True)
    concatenated_df.to_csv(output_csv_path, index=False)


if __name__ == '__main__':
    input_excel_path = pl.Path("FIA Salida 2024 - Cuaja.xlsx")
    output_csv_path = pl.Path("salida_cuaja.csv")
    excel_to_csv(input_excel_path, output_csv_path)
