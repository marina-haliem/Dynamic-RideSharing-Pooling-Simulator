# coding: utf-8

# # Experimental Results
# This script parses experimental logs to obtain performance metrics.
# Please note that log_analyzer.py is used from the tools directory.
# Documentation of the usage for the LogAnalyzer class is provided on log_analyzer.py

# Import LogAnalyzer objects
from tools.log_analyzer import *
from config.settings import DEFAULT_LOG_DIR

# and other relevant stuff...
import matplotlib.pyplot as plt

# #### !IMPORTANT: Specify directory and log filenames here 
# Note that the provided names (below) are default names. They do not have to be changes unless you decided to rename files from multiple experiments.
vehicle_log_file = "/sim/vehicle.log"
customer_log_file = "/sim/customer.log"
score_log_file = "/sim/score.log"
summary_log_file = "/sim/summary.log"


# Invoke LogAnalyzer Object
l = LogAnalyzer()


# #### Exploring dataframes of each of the logs
# Loading all the different logs as pandas dataframes
summary_df = l.load_summary_log(DEFAULT_LOG_DIR)
vehicle_df = l.load_vehicle_log(DEFAULT_LOG_DIR)
customer_df = l.load_customer_log(DEFAULT_LOG_DIR)
score_df = l.load_score_log(DEFAULT_LOG_DIR)

print(summary_df.describe())

print(vehicle_df.describe())

print(customer_df["waiting_time"].describe())

print(score_df.describe())


# #### Exploring the get_customer_status
df = l.get_customer_status(customer_df)
print(df.head())

df = l.get_customer_waiting_time(customer_df)
print(df.head())


##### Generating plots of summary logs
## You can have more than one experiment that you want to compare, you can pass their relevant paths here to the first
## list argument
summary_plots = l.plot_summary([DEFAULT_LOG_DIR], ["Number of Accepted Requests", "Average Travel Distance",
												   "Occupancy Rate of Vehicles"], plt)
summary_plots.savefig(DEFAULT_LOG_DIR+"/Summary.pdf", bbox_inches = 'tight')
summary_plots.show()

##### Generating plots of relevant experiment metrics
plt, df = l.plot_metrics([DEFAULT_LOG_DIR], ["Profit", "Cruising Time", "Occupancy Rate","Waiting Time", "Travel Distance"],plt)
plt.savefig(DEFAULT_LOG_DIR+"/Metrics.pdf", bbox_inches = 'tight')
# plt.show()

# #### We may also look at the metrics as a pandas dataframe
print(df.head())
