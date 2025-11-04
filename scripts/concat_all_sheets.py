import argparse
import logging
import sys

import pandas as pd
import pathlib as pl

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input excel file path')
    parser.add_argument('--output', type=str, required=True, help='output file path')
    return parser

def create_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

def concat_all_sheets(input_file: pl.Path) -> pd.DataFrame:
    xl = pd.ExcelFile(input_file)
    df_list = []
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        df_list.append(df)

    # Make every df columns lower
    for i in range(len(df_list)):
        df_list[i].columns = df_list[i].columns.str.lower()

    # Check every sheet has the same columns
    first_columns = df_list[0].columns
    for i, df in enumerate(df_list[1:], start=1):
        if not df.columns.equals(first_columns):
            raise ValueError(f"Sheet {xl.sheet_names[i]} has different columns than the first sheet.")

    concatenated_df = pd.concat(df_list, ignore_index=True)
    return concatenated_df

def main():
    logger = create_logger()
    parser = create_parser()
    args   = parser.parse_args()

    input_file       = pl.Path(args.input)
    output_file      = pl.Path(args.output)
    output_extension = output_file.suffix.lower()

    if not input_file.exists():
        raise FileNotFoundError(f"Input file {input_file} does not exist or is not an excel file.")

    if input_file.suffix not in ['.xls', '.xlsx']:
        raise ValueError(f"Input file {input_file} is not an excel file.")

    xl = pd.ExcelFile(input_file)
    df_list = []
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        df_list.append(df)
        logger.info(f"Read sheet: {sheet_name} with shape {df.shape}")

    concatenated_df = concat_all_sheets(input_file)
    logger.info(f"Concatenated shape: {concatenated_df.shape}")

    if output_extension == '.csv':
        concatenated_df.to_csv(output_file, index=False)
    else:
        raise NotImplementedError("Only CSV output is currently supported.")

    logger.info(f"Saved concatenated data to {output_file}")


if __name__ == '__main__':
    main()




