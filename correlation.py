from pandas.plotting import scatter_matrix
from utils.manipulate_data import get_cleaned_data
import matplotlib.pyplot as plt


df = get_cleaned_data()

attributes = ["input_rows_quantity","output_rows_quantity","number_of_workers", "photon_acceleration", "constraint","cte","case_when","inner_join","left_join","right_join","group_by","where","and","or","subquery", "vcpu", "memory_ram_gb", "instance_storage_type"]
scatter_matrix(df[attributes], figsize=(15, 10), color='#840E6B', hist_kwds={'color':['#A029FA']})

plt.show()