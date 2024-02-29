import numpy as np

def columns_to_be_used_as_input():
    return ["input_rows_quantity", "input_columns_quantity","output_columns_quantity","number_of_workers", "photon_acceleration", "constraint","cte","case_when","inner_join","left_join","right_join","group_by", "selectivity_factor","subquery", "explode", "vcpu", "memory_ram_gb", "instance_storage_type"]

def column_to_be_used_as_output():
    return ["runtime_range"]

def format_data(df):
    return df.astype({"input_rows_quantity":"int","input_columns_quantity":"int","output_columns_quantity":"int","number_of_workers":"int","constraint":"int","cte":"int","case_when":"int","inner_join":"int","left_join":"int","right_join":"int","group_by":"int","selectivity_factor":"float","subquery":"int", "vcpu":"int", "memory_ram_gb":"int"})

def encode_columns(df, columns_encoders):
    for value in columns_encoders:
        df[value.get("column")] = value.get("encoder").fit_transform(df[value.get("column")])
    return df

def get_cleaned_data(df, columns_encoders, columns_to_be_used_as_input, column_to_be_used_as_output, number_of_ranges):
    
    if number_of_ranges == 8:
        df['runtime_range'] = np.where(df['runtime'] <= 30, '0 to 30', #0 a 30s
                            np.where(df['runtime'] <= 300, '30 to 300', #30s to 5m
                            np.where(df['runtime'] <= 600, '300 to 600', #5m to 10m
                            np.where(df['runtime'] <= 1800, '600 to 1800', #10m to 30m
                            np.where(df['runtime'] <= 3600, '1800 to 3600', #30m to 60m
                            np.where(df['runtime'] <= 7200, '3600 to 7200', #60m to 2h
                            np.where(df['runtime'] <= 14400, '7200 to 14400', '> 14400'))))))) #2h to 4h

    if number_of_ranges == 7:
        df['runtime_range'] = np.where(df['runtime'] <= 300, '0 to 300',
                            np.where(df['runtime'] <= 600, '300 to 600',
                            np.where(df['runtime'] <= 1800, '600 to 1800',
                            np.where(df['runtime'] <= 3600, '1800 to 3600',
                            np.where(df['runtime'] <= 7200, '3600 to 7200',
                            np.where(df['runtime'] <= 14400, '7200 to 14400', '> 14400'))))))

    if number_of_ranges == 6:
        df['runtime_range'] = np.where(df['runtime'] <= 300, '0 to 300',
                        np.where(df['runtime'] <= 1800, '300 to 1800',
                        np.where(df['runtime'] <= 3600, '1800 to 3600',
                        np.where(df['runtime'] <= 7200, '3600 to 7200',
                        np.where(df['runtime'] <= 14400, '7200 to 14400', '> 14400')))))

    df = df[columns_to_be_used_as_input + column_to_be_used_as_output]
    df = format_data(df)
    df = encode_columns(df, columns_encoders)

    return df
