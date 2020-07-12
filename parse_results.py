# coding: utf-8

# # Experimental Results
# This script parses experimental logs to obtain performance metrics.
# Please note that log_analyzer.py is used from the tools directory.
# Documentation of the usage for the LogAnalyzer class is provided on log_analyzer.py

# Import LogAnalyzer objects
from tools.log_analyzer import *

# and other relevant stuff...
import matplotlib.pyplot as plt

# #### !IMPORTANT: Specify directory and log filenames here 
# Note that the provided names (below) are default names. They do not have to be changes unless you decided to rename files from multiple experiments.
log_dir_path = "./baselines/DARM/sim/"
vehicle_log_file = "vehicle.log"
customer_log_file = "customer.log"
score_log_file = "score.log"
summary_log_file = "summary.log"


# Invoke LogAnalyzer Object
l = LogAnalyzer()


# #### Exploring dataframes of each of the logs
# Loading all the different logs as pandas dataframes
summary_df = l.load_summary_log(log_dir_path)
vehicle_df = l.load_vehicle_log(log_dir_path)
customer_df = l.load_customer_log(log_dir_path)
score_df = l.load_score_log(log_dir_path)

print(summary_df.describe())

print(vehicle_df.describe())

print(customer_df["waiting_time"].describe())

print(score_df.describe())


# #### Exploring the get_customer_status
df = l.get_customer_status(customer_df)
print(df.head())

df = l.get_customer_waiting_time(customer_df)
print(df.head())


# #### Generating plots of summary logs
DARM = "./baselines/DARM/sim/"
DPRS = "./baselines/DPRS/sim/"
Deep_Pool = "./baselines/Deep_Pool/sim/"
ODA = "./baselines/ODA/sim/"
Dummy_RS = "./baselines/Dummy_RS/sim/"
Dummy_None = "./baselines/Dummy_None/sim/"

# summary_plots = l.plot_summary([DARM, DPRS, Deep_Pool, ODA, Dummy_RS, Dummy_None], ["Number of Accepted Requests",
# 																"Number of Rejected Requests", "Occupancy "
# 																										"Rate of "
# 																							   "Vehicles"], plt)
# summary_plots.savefig("./baselines/DARM/Summary.pdf", bbox_inches = 'tight')
# summary_plots.show()

# #### Generating plots of relevant experiment metrics
plt, df = l.plot_metrics([log_dir_path], ["Profit", "Cruising Time", "Occupancy Rate","Waiting Time", "Travel "
																								"Distance"],plt)
plt.savefig("./baselines/DARM/Metrics.pdf", bbox_inches = 'tight')
plt.show()

# #### We may also look at the metrics as a pandas dataframe
print(df.head())
