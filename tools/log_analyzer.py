import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from simulator.settings import WORKING_COST, DRIVING_COST
from common import time_utils
from config import settings
import matplotlib.dates as md
import datetime as dt
import matplotlib.ticker as tkr
from config.settings import DEFAULT_LOG_DIR
from matplotlib.ticker import MaxNLocator
import numpy as np

log_dir_path = DEFAULT_LOG_DIR
vehicle_log_file = "/sim/vehicle.log"
customer_log_file = "/sim/customer.log"
score_log_file = "/sim/score.log"
summary_log_file = "/sim/summary.log"

vehicle_log_cols = [
    "t",
    "id",
    "lat",
    "lon",
    "speed",
    "status",
    "destination_lat",
    "destination_lon",
    "type",
    "travel_dist",
    "price_per_travel_m",
    "price_per_wait_min",
    "gas_price",
    "assigned_customer_id",
    "time_to_destination",
    "idle_duration",
    "current_capacity",
    "max_capacity",
    "driver_base_per_trip",
    "mileage",
    "agent_type",
]

customer_log_cols = ["t", "id", "status", "waiting_time"]

summary_log_cols = [
    "readable_time",  ####Human readable time
    "t",  ####Unix time
    "n_vehicles_OnDuty",  ####On duty vehicles
    "n_vehicles_Occupied",  ####Fully occupied vehicles
    "n_requests",  ####Total number of requests
    "n_requests_assigned",  ####Number of requests assigned
    "n_rejected_requests",  ####Number of Rejected Requests
    "n_accepted_commands",  ####Number of Requests Accepted by customers
    "average_wt",  ####Average Wait for all customers
    "avg_earnings",  ####Average Earnings per vehicle
    "avg_cost",  ####Average Cost per vehicle
    "avg_profit_dqn",  ####Average Profit per vehicle using dqn agent
    "avg_profit_dummy",  ####Average Profit per vehicle using dummy agent
    "avg_total_dist",  ####Average total distance travelled by vehicles (dqn or dummy?)
    "avg_cap",  ####Average SEATS occupancy per vehicle
    "avg_idle_time",  ####Average idle time per vehicle
]

score_log_cols = [
    "t",
    "vehicle_id",
    "travel_dist",
    "profit",
    "working_time",
    "earning",
    "idle",
    "cruising",
    "occupied",
    "assigned",
    "offduty",
]


class LogAnalyzer(object):
    """This class helps analyze experimental results.
    The simulator should have output '.log' files in a given logging directory.
    The following log files may be parsed by this LogAnalyzer:
    summary.log, vehicle.log, customer.log, score.log
    """

    def __init__(self):
        pass

    def load_log(self, path, cols, max_num, skip_minutes=0):
        """Parses .log files into pd.DataFrame
        Args:
            path:           (str) Directory where sim logs are present.
            cols:           (list) List of column names for a given log file.
            max_num:        (int) Max number of logs to parse (if multiple experiments were run).
            skip_minutes:   (int) Number of minutes to skip in logs (from the top).
        Returns:
            df:             (pd.DataFrame) Logs are now returned as a DataFrame for easy manipulation.
        """
        df = pd.read_csv(path, names=cols)
        dfs = [df]
        for i in range(1, max_num):
            # pdb.set_trace()
            path_ = path + "." + str(i)
            try:
                df = pd.read_csv(path_, names=cols)
                dfs.append(df)
            except IOError:
                break
        df = pd.concat(dfs)
        df = df[df.t >= df.t.min() + skip_minutes * 60]
        return df

    def load_vehicle_log(self, log_dir_path, max_num=100, skip_minutes=0):
        """Used to obtain vehicle logs as a DataFrame"""
        return self.load_log(
            log_dir_path + vehicle_log_file, vehicle_log_cols, max_num, skip_minutes
        )

    def load_customer_log(self, log_dir_path, max_num=100, skip_minutes=0):
        """Used to obtain customer logs as a DataFrame"""
        return self.load_log(
            log_dir_path + customer_log_file, customer_log_cols, max_num, skip_minutes
        )

    def load_summary_log(self, log_dir_path, max_num=100, skip_minutes=0):
        print("Summary: ", log_dir_path + summary_log_file)
        """Used to obtain summary logs as a DataFrame"""
        return self.load_log(
            log_dir_path + summary_log_file, summary_log_cols, max_num, skip_minutes
        )

    def _load_score_log(self, log_dir_path, max_num=100, skip_minutes=0):
        """Used as a helper function for load_score_log"""
        return self.load_log(
            log_dir_path + score_log_file, score_log_cols, max_num, skip_minutes
        )

    def load_score_log(self, log_dir_path, max_num=100, skip_minutes=0):
        """Used to obtain score logs as a DataFrame"""
        df = self._load_score_log(log_dir_path, max_num, skip_minutes)
        # total_seconds = (df.t.max() - df.t.min() + 3600 * 24)
        # n_days = total_seconds / 3600 / 24
        # df = df[df.t == df.t.max()]
        # df["working_hour"] = (total_seconds - df.offduty) / n_days / 3600
        df["working_hour"] = (df.working_time - df.offduty) / 3600
        df["cruising_hour"] = (df.cruising + df.assigned) / 3600
        df["occupancy_rate"] = df.occupied / (df.working_hour * 3600) * 100
        df["reward"] = (
            df.earning
            - (df.cruising + df.assigned + df.occupied)
            * DRIVING_COST
            / settings.TIMESTEP
            - (df.working_time - df.offduty) * WORKING_COST / settings.TIMESTEP
        )
        df["revenue_per_hour"] = df.earning / df.working_hour
        df["profit_per_hour"] = df.profit / df.working_hour

        return df

    def get_customer_status(self, customer_df, bin_width=300):
        """Customer Status (discretized by time0"""
        customer_df["time_bin"] = self.add_time_bin(customer_df, bin_width)
        df = (
            customer_df.groupby(["time_bin", "status"])
            .size()
            .reset_index()
            .pivot(index="time_bin", columns="status", values=0)
            .fillna(0)
        )
        df = df.rename(columns={2: "ride_on", 4: "rejected"})
        # TODO maybe re write this
        df["total"] = sum([x for _, x in df.items()])
        df.index = [time_utils.get_local_datetime(x) for x in df.index]
        return df

    def get_customer_waiting_time(self, customer_df, bin_width=300):
        """Customer Waiting time (discretized time)"""
        customer_df["time_bin"] = self.add_time_bin(customer_df, bin_width)
        df = (
            customer_df[customer_df.status == 2].groupby("time_bin").waiting_time.mean()
        )
        df.index = [time_utils.get_local_datetime(x) for x in df.index]
        return df

    def add_time_bin(self, df, bin_width):
        """Helper function to discretize time from minutes into bins of 'bin_width'"""
        start_time = df.t.min()
        return ((df.t - start_time) / bin_width).astype(int) * int(
            bin_width
        ) + start_time

    def numfmt(self, x, pos):
        if int(x / 1000) == 0:
            s = "{}".format(x / 1000.0)
        else:
            s = "{}".format(int(x / 1000))
        return s

    def plot_summary(self, paths, labels, plt):
        """Plotting of experiment summaries
        Args:
            paths:      (list) List of paths of all experiments which are to be plotted.
            labels:     (list) Names for each of the respective experiments.
            plt:        (matplotlib.pyplot) matplotlib object to write the plot onto???
        Returns:
            plt:        (matplotlib.pyplot) The output plot.
        """
        plt.figure(figsize=(18, 5))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        for i, path in enumerate(paths):
            summary = self.load_summary_log(path)
            if i == 0:
                print("Size: ", len(summary))
                summary = pd.concat([summary, summary])
                print(len(summary))
            # print(summary)
            summary["t"] = (summary.t / 3600).astype(int) * 3600
            # summary["t"] = summary.t/60
            summary = summary.groupby("t").mean().reset_index()
            # summary.t = [time_utils.get_local_datetime(t) for t in summary.t]
            # print(summary.t)
            summary.t = [dt.datetime.fromtimestamp(t + 10) for t in summary.t]
            summary["day"] = [t.day for t in summary.t]
            years = md.YearLocator()  # every year
            months = md.MonthLocator()  # every month
            days = md.DayLocator()
            hrs = md.HourLocator()
            xfmt = md.DateFormatter("| %a |")

            # Labels to add to the legend for each algorithm to compare
            if i == 0:
                l = "DARM + DPRS"

            plt.subplot(131)
            # plt.subplot(len(paths), 3, +i * 2 + 1)
            # print(summary.t)
            summary["rate"] = summary.n_accepted_commands / summary.n_requests
            plt.plot(summary.t, summary.n_accepted_commands, label=l)

            # if i == 0:
            #     plt.plot(summary.t, summary.n_requests, linestyle=":")

            # plt.plot(
            #     summary.t, summary.n_rejected_requests, label="Reject", linestyle=":"
            # )

            # plt.plot(
            #     summary.t,(summary.n_requests - summary.n_requests_assigned), label="Rejected by System", linestyle=":"
            # )
            #
            # plt.plot(
            #     summary.t, summary.n_rejected_requests - (summary.n_requests - summary.n_requests_assigned),
            #     label="Rejected by Customers", linestyle=":"
            # )
            #
            # plt.plot(summary.t, summary.n_accepted_commands, label="Accepted", alpha=0.7)

            plt.ylabel("Accept Rate / hour")
            plt.title(labels[0])
            # plt.xlabel("simulation time (yy-mm-dd hr:min:sec)")
            plt.xlabel("Simulation Time (hrs in days)")
            plt.xticks(ha="left")
            # plt.ylim([0, 610])
            ax = plt.gca()
            ax.set_xticks(summary.t)
            ax.xaxis.set_major_locator(days)
            # ax.xaxis.set_major_locator(MaxNLocator(prune='both'))
            ax.xaxis.set_minor_locator(hrs)
            ax.xaxis.set_major_formatter(xfmt)
            plt.legend(loc="lower right", framealpha=0.7)

            plt.subplot(133)
            # plt.subplot(len(paths), 3, i * 2 + 2)
            plt.title(labels[2])
            plt.plot(summary.t, summary.n_vehicles_Occupied, label=l)
            # plt.plot(
            #     summary.t, summary.n_vehicles_Occupied, label="Occupied", linestyle=":"
            # )
            yfmt = tkr.FuncFormatter(self.numfmt)
            plt.gca().yaxis.set_major_formatter(yfmt)
            plt.ylabel("# of occupied vehicles (in 1000s) per hour")
            # plt.ylabel("# of vehicles")
            plt.xlabel("Simulation Time (hrs in days)")
            # plt.xlabel("simulation time (yy-mm-dd hh:min:sec)")
            plt.xticks(ha="left")
            # plt.ylim([0, 10100])
            ax = plt.gca()
            ax.set_xticks(summary.t)
            ax.xaxis.set_major_locator(days)
            ax.xaxis.set_minor_locator(hrs)
            ax.xaxis.set_major_formatter(xfmt)
            # print(ax.get_xticklabels()[0])
            # if i != len(paths) - 1:
            #     plt.xticks([])
            # if i == 0:
            plt.legend(loc="lower right", framealpha=0.7)

            plt.subplot(132)
            # plt.subplot(len(paths), 3, i * 2 + 2)
            plt.title(labels[1])
            plt.plot(summary.t, summary.avg_total_dist, label=l)
            # plt.plot(summary.t, (summary.n_requests - summary.n_requests_assigned), label=l)
            # plt.plot(
            #     summary.t, summary.n_vehicles_Occupied, label="Reject", linestyle=":"
            # )
            plt.ylabel("Avg. Travel Distance per vehicle per hour (in Km)")
            plt.xlabel("Simulation Time (hrs in days)")
            # plt.xlabel("simulation time (yy-mm-dd hh:min:sec)")
            plt.xticks(ha="left")
            # plt.ylim([0, 10100])
            ax = plt.gca()
            ax.set_xticks(summary.t)
            ax.xaxis.set_major_locator(days)
            ax.xaxis.set_minor_locator(hrs)
            ax.xaxis.set_major_formatter(xfmt)
            # print(ax.get_xticklabels()[0])
            # if i != len(paths) - 1:
            #     plt.xticks([])
            # if i == 0:
            plt.legend(loc="upper left", framealpha=0.7)
        return plt

    def plot_metrics_ts(self, paths, labels, plt):
        """Plotting of experiment Scores
        Args:
            paths:      (list) List of paths of all experiments which are to be plotted.
            labels:     (list) Names for each of the respective experiments.
            plt:        (matplotlib.pyplot) matplotlib object to write the plot onto???
        Returns:
            plt:        (matplotlib.pyplot) The output plot
        """
        plt.figure(figsize=(12, 4))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        for p, label in zip(paths, labels):
            score = self.load_score_log(p)
            score["t"] = ((score.t - score.t.min()) / 3600).astype(int)
            plt.subplot(131)
            plt.ylabel("revenue ($/h)")
            plt.scatter(score.t, score.revenue_per_hour, alpha=0.5, label=label)
            plt.ylim([0, 50])
            plt.subplot(132)
            plt.ylabel("cruising time (h/day)")
            plt.scatter(score.t, score.cruising_hour, alpha=0.5, label=label)

        plt.legend()
        return plt

    def plot_metrics(self, paths, labels, plt):
        """Plotting of misc. experiment metrics (uses vehicle, customer, and score logs.)
        Args:
            paths:      (list) List of paths of all experiments which are to be plotted.
            labels:     (list) Names for each of the respective experiments.
            plt:        (matplotlib.pyplot) matplotlib object to write the plot onto???
        Returns:
            plt:        (matplotlib.pyplot) The output plot.
        """
        data = []
        yfmt = tkr.FuncFormatter(self.numfmt)
        plt.figure(figsize=(15, 3))
        plt.subplots_adjust(wspace=0.3, hspace=0.4)
        for p, label in zip(paths, labels):
            score = self.load_score_log(p)
            c = self.load_customer_log(p, skip_minutes=60)

            plt.subplot(151)
            plt.xlabel("Profit ($/h)")
            plt.hist(
                score.profit_per_hour,
                bins=100,
                range=((score.profit_per_hour.min() + 50), score.profit_per_hour.max()),
                alpha=0.5,
                label=labels[0],
            )
            # plt.yticks([])
            plt.gca().yaxis.set_major_formatter(yfmt)
            plt.ylabel("# of vehicles (in 1000s)")
            plt.title(labels[0])

            # plt.subplot(172)
            # plt.xlabel("working time (h/day)")
            # plt.hist(score.working_hour, bins=100, range=(score.working_hour.min(), score.working_hour.max()),
            #          alpha=0.5, label=labels[1])
            # # plt.yticks([])
            # plt.gca().yaxis.set_major_formatter(yfmt)
            # plt.ylabel("# of vehicles (in 1000s)")
            # plt.title(labels[1])

            plt.subplot(152)
            plt.xlabel("Cruising time (h/day)")
            plt.hist(
                score.cruising_hour,
                bins=100,
                range=(score.cruising_hour.min(), score.cruising_hour.max()),
                alpha=0.5,
                label=labels[1],
            )
            # plt.yticks([])
            plt.gca().yaxis.set_major_formatter(yfmt)
            plt.ylabel("# of vehicles (in 1000s)")
            plt.title(labels[1])

            plt.subplot(153)
            plt.xlabel("Occupancy Rate (%)")
            plt.hist(
                score.occupancy_rate,
                bins=100,
                range=(score.occupancy_rate.min(), score.occupancy_rate.max()),
                alpha=0.5,
                label=labels[2],
            )
            # plt.yticks([])
            plt.gca().yaxis.set_major_formatter(yfmt)
            plt.ylabel("# of vehicles (in 1000s)")
            plt.title(labels[2])
            #
            # plt.subplot(235)
            # plt.xlabel("total reward / day")
            # plt.hist(score.reward, bins=100, range=(-10, 410), alpha=0.5, label=label)
            # plt.yticks([])

            plt.subplot(155)
            plt.xlabel("Waiting Time (sec.)")
            plt.hist(
                c[c.status == 2].waiting_time,
                bins=500,
                range=(c[c.status == 2].waiting_time.min(), 650),
                alpha=0.5,
                label=labels[3],
            )
            # plt.yticks([])
            plt.gca().yaxis.set_major_formatter(yfmt)
            plt.ylabel("# of customers (in 1000s)")
            plt.title(labels[3])

            plt.subplot(154)
            plt.xlabel("Travel Distance (km/h)")
            plt.gca().xaxis.set_major_formatter(yfmt)
            score.travel_dist /= score.working_hour
            plt.hist(
                score.travel_dist,
                bins=100,
                range=(score.travel_dist.min(), score.travel_dist.max()),
                alpha=0.5,
                label=labels[4],
            )
            # plt.yticks([])
            plt.gca().yaxis.set_major_formatter(yfmt)
            plt.ylabel("# of vehicles (in 1000s)")
            plt.title(labels[4])

            # plt.subplot(177)
            # plt.xlabel("Profit (in $/h)")
            # plt.hist(score.profit_per_hour, bins=100, range=(score.profit_per_hour.min(), score.profit_per_hour.max()),
            #          alpha=0.5, label=labels[6])
            # # plt.yticks([])
            # plt.gca().yaxis.set_major_formatter(yfmt)
            # plt.ylabel("# of vehicles (in 1000s)")
            # plt.title(labels[6])

            x = {}
            x["00_reject_rate"] = float(len(c[c.status == 4])) / len(c) * 100
            x["01_revenue/hour"] = score.revenue_per_hour.mean()
            x["02_occupancy_rate"] = score.occupancy_rate.mean()
            x["03_cruising/day"] = score.cruising_hour.mean()
            x["04_working/day"] = score.working_hour.mean()
            x["05_waiting_time"] = c[c.status == 2].waiting_time.mean()

            x["11_revenue/hour(std)"] = score.revenue_per_hour.std()
            x["12_occupancy_rate(std)"] = score.occupancy_rate.std()
            x["13_cruising/day(std)"] = score.cruising_hour.std()
            x["14_working/day(std)"] = score.working_hour.std()
            x["15_waiting_time(std)"] = c[c.status == 2].waiting_time.std()
            data.append(x)

        # plt.legend()

        df = pd.DataFrame(data, index=labels)
        return plt, df
