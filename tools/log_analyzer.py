import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../src/')
from simulator.settings import WORKING_COST, DRIVING_COST
from common import time_utils
from config import settings

log_dir_path = "data/logs/"
vehicle_log_file = "vehicle.log"
customer_log_file = "customer.log"
score_log_file = "score.log"
summary_log_file = "summary.log"

vehicle_log_cols = [
    "t",
    "id",
    "lat",
    "lon",
    "speed",
    "status",
    "destination_lat",
    "destination_lon",
    "assigned_customer_id",
    "time_to_destination",
    "idle_duration"
    ]

customer_log_cols = [
    "t",
    "id",
    "status",
    "waiting_time"
]

summary_log_cols = [
    "t",
    "n_vehicles",
    "occupied_vehicles",
    "n_requests",
    "n_matching",
    "n_dispatch",
    "average_wt"
]

score_log_cols = [
    "t",
    "vehicle_id",
    "working_time",
    "earning",
    "idle",
    "cruising",
    "occupied",
    "assigned",
    "offduty"
]

class LogAnalyzer(object):

    def __init__(self):
        pass

    def load_log(self, path, cols, max_num, skip_minutes=0):
        df = pd.read_csv(path, names=cols)
        dfs = [df]
        for i in range(1, max_num):
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
        return self.load_log(log_dir_path + vehicle_log_file, vehicle_log_cols, max_num, skip_minutes)

    def load_customer_log(self, log_dir_path, max_num=100, skip_minutes=0):
        return self.load_log(log_dir_path + customer_log_file, customer_log_cols, max_num, skip_minutes)

    def load_summary_log(self, log_dir_path, max_num=100, skip_minutes=0):
        return self.load_log(log_dir_path + summary_log_file, summary_log_cols, max_num, skip_minutes)

    def _load_score_log(self, log_dir_path, max_num=100, skip_minutes=0):
        return self.load_log(log_dir_path + score_log_file, score_log_cols, max_num, skip_minutes)

    def load_score_log(self, log_dir_path, max_num=100, skip_minutes=0):
        df = self._load_score_log(log_dir_path, max_num, skip_minutes)
        # total_seconds = (df.t.max() - df.t.min() + 3600 * 24)
        # n_days = total_seconds / 3600 / 24
        # df = df[df.t == df.t.max()]
        # df["working_hour"] = (total_seconds - df.offduty) / n_days / 3600
        df["working_hour"] = (df.working_time - df.offduty) / 3600
        df["cruising_hour"] = (df.cruising + df.assigned) / 3600
        df["occupancy_rate"] = df.occupied / (df.working_hour * 3600) * 100
        df["reward"] = (df.earning
                       - (df.cruising + df.assigned + df.occupied) * DRIVING_COST / settings.TIMESTEP
                       - (df.working_time - df.offduty) * WORKING_COST / settings.TIMESTEP)
        df["revenue_per_hour"] = df.earning / df.working_hour

        return df

    def get_customer_status(self, customer_df, bin_width=300):
        customer_df["time_bin"] = self.add_time_bin(customer_df, bin_width)
        df = customer_df.groupby(["time_bin", "status"]).size().reset_index().pivot(index="time_bin", columns="status", values=0).fillna(0)
        df = df.rename(columns={2 : "ride_on", 4 : "rejected"})
        df["total"] = sum([x for _, x in df.iteritems()])
        df.index = [time_utils.get_local_datetime(x) for x in df.index]
        return df

    def get_customer_waiting_time(self, customer_df, bin_width=300):
        customer_df["time_bin"] = self.add_time_bin(customer_df, bin_width)
        df = customer_df[customer_df.status == 2].groupby("time_bin").waiting_time.mean()
        df.index = [time_utils.get_local_datetime(x) for x in df.index]
        return df

    def add_time_bin(self, df, bin_width):
        start_time = df.t.min()
        return ((df.t - start_time) / bin_width).astype(int) * int(bin_width) + start_time


    def plot_summary(self, paths, labels, plt):
        plt.figure(figsize=(12, 5))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        for i, path in enumerate(paths):
            summary = self.load_summary_log(path)
            summary['t'] = (summary.t / 3600).astype(int) * 3600
            summary = summary.groupby('t').mean().reset_index()
            summary.t = [time_utils.get_local_datetime(t) for t in summary.t]

            plt.subplot(len(paths), 2, + i * 2 + 1)
            plt.plot(summary.t, summary.n_requests, label="request")
            plt.plot(summary.t, summary.n_requests - summary.n_matching, label="reject", linestyle=':')
            plt.plot(summary.t, summary.n_dispatch, label="dispatch", alpha=0.7)
            plt.ylabel("count/minute")
            plt.ylim([0, 610])
            if i != len(paths) - 1:
                plt.xticks([])
            if i == 0:
                plt.legend(loc='upper right')

            plt.subplot(len(paths), 2, i * 2 + 2)
            plt.title(labels[i])
            plt.plot(summary.t, summary.n_vehicles, label="working")
            plt.plot(summary.t, summary.occupied_vehicles, label="occupied", linestyle=':')
            plt.ylabel("# of vehicles")
            plt.ylim([0, 10100])
            if i != len(paths) - 1:
                plt.xticks([])
            if i == 0:
                plt.legend(loc='upper right')
                # plt.subplot(313)
            # plt.plot(summary.t, summary.average_wt, alpha=1.0)
            # plt.ylim([0, 450])
            # plt.ylabel("waiting time (s)")
            # plt.xlabel("simulation time (s)")
        return plt

    def plot_metrics_ts(self, paths, labels, plt):
        plt.figure(figsize=(12, 4))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        for p, label in zip(paths, labels):
            score = self.load_score_log(p)
            score['t'] = ((score.t - score.t.min()) / 3600).astype(int)
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
        data = []
        plt.figure(figsize=(12, 3))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        for p, label in zip(paths, labels):
            score = self.load_score_log(p)
            c = self.load_customer_log(p, skip_minutes=60)

            plt.subplot(141)
            plt.xlabel("revenue ($/h)")
            plt.hist(score.revenue_per_hour, bins=100, range=(18, 42), alpha=0.5, label=label)
            plt.yticks([])

            plt.subplot(142)
            plt.xlabel("working time (h/day)")
            plt.hist(score.working_hour, bins=100, range=(17, 23), alpha=0.5, label=label)
            plt.yticks([])

            plt.subplot(143)
            plt.xlabel("cruising time (h/day)")
            plt.hist(score.cruising_hour, bins=100, range=(1.9, 7.1), alpha=0.5, label=label)
            plt.yticks([])

            # plt.subplot(234)
            # plt.xlabel("occupancy rate")
            # plt.hist(score.occupancy_rate, bins=100, range=(55, 75), alpha=0.5, label=label)
            # plt.yticks([])
            #
            # plt.subplot(235)
            # plt.xlabel("total reward / day")
            # plt.hist(score.reward, bins=100, range=(-10, 410), alpha=0.5, label=label)
            # plt.yticks([])

            plt.subplot(144)
            plt.xlabel("waiting time (s)")
            plt.hist(c[c.status==2].waiting_time, bins=500, range=(0, 650), alpha=0.5, label=label)
            plt.yticks([])

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

        plt.legend()
        df = pd.DataFrame(data, index=labels)
        return plt, df
