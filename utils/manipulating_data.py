import pandas as pd
from sklearn import preprocessing

def add_new_columns():
    data = [[0, 100], [100, 200], [200, 400], [400, 600], [600, 800], 
    [1000, 1500], [1500, 2000], [2000, 3000],[3000, 4000], [4000, 5000], 
    [5000, 6000], [6000, 7000], [7000, 8000], [8000, 9000], [9000, 10000], 
    [10000, 15000], [15000, 20000], [20000, 30000]]
  
    return pd.DataFrame(data, columns=['Name', 'runtime_range'])

def columns_to_be_used_as_input():
    return ["input_rows_quantity", "input_columns_quantity","output_columns_quantity","number_of_workers", "photon_acceleration", "constraint","cte","case_when","inner_join","left_join","right_join","group_by", "selectivity_factor","subquery", "explode", "create", "read", "update", "delete", "vcpu", "memory_ram_gb", "instance_storage_type"]

# ter um fator de seletividade - pegar da estatística do banco -> só tem quantidade de bytes nas estatisticas
# databricks: analyse table - compute statistics -> mesma respota da pergunta acima
# tem agregações? count, avg, first, last etc
# funções escalares talvez não precisem (lower, upper etc)
# a query tem index?
# qtd de colunas de entrada e saída
# pesquisar se é possivel caracterizar uma consulta sql

def column_to_be_used_as_output():
    return ["runtime"]

def get_cleaned_data(columns_to_be_used_as_input, column_to_be_used_as_output):

    # INPUT DATA
    executions = pd.read_csv("data/input/query_executions.csv",thousands=',')
    operations = pd.read_csv("data/input/query_operations.csv", thousands=',')
    instances = pd.read_csv("data/input/query_instances.csv", thousands=',')

    # QUERY
    df = executions.merge(instances, left_on='worker_type', right_on='instance_type', how="left").merge(operations, on='query_name', how="left")
    df = df[columns_to_be_used_as_input + column_to_be_used_as_output]

    # FORMAT
    df = df.astype({"input_rows_quantity":"int","input_columns_quantity":"int","output_columns_quantity":"int","number_of_workers":"int","constraint":"int","cte":"int","case_when":"int","inner_join":"int","left_join":"int","right_join":"int","group_by":"int","selectivity_factor":"float","subquery":"int", "vcpu":"int", "memory_ram_gb":"int"})

    le = preprocessing.LabelEncoder()
    df['instance_storage_type'] = le.fit_transform(df['instance_storage_type'])
    df['photon_acceleration'] = le.fit_transform(df['photon_acceleration'])
    df['create'] = le.fit_transform(df['create'])
    df['read'] = le.fit_transform(df['read'])
    df['update'] = le.fit_transform(df['update'])
    df['delete'] = le.fit_transform(df['delete'])

    return df
