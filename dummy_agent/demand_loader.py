import numpy as np
import pandas as pd
from db import engine
from common.time_utils import get_local_datetime
from config.settings import MAP_WIDTH, MAP_HEIGHT, GLOBAL_STATE_UPDATE_CYCLE,\
    DESTINATION_PROFILE_TEMPORAL_AGGREGATION, DESTINATION_PROFILE_SPATIAL_AGGREGATION


class DemandLoader(object):
    def __init__(self, timestep=1800, amplification_factor=1.0):
        self.timestep = timestep
        self.amplification_factor = amplification_factor
        self.current_time = None
        self.hourly_demand = []

    # Loading demand within time t
    def load(self, t, horizon=2):
        x = self.update_hourly_demand(t - self.timestep)
        demand = []

        for _ in range(horizon + 1):
            if abs(x) <= 0.5:   # If less than 30 minutes
                d = self.__compute_demand(x, self.hourly_demand[0:2])
            elif 0.5 < x and x <= 1.5:  # If > half an hour and < 1.5 hours
                d = self.__compute_demand(x - 1, self.hourly_demand[1:3])
            elif 1.5 < x and x <= 2.5:
                d = self.__compute_demand(x - 2, self.hourly_demand[2:4])
            else:
                raise NotImplementedError

            x += self.timestep / 3600.0
            demand.append(d)

        # Loading demand of the latest timestep
        latest_demand = self.load_latest_demand(t - self.timestep, t)
        # Return demand in all timesteps 1 to current and the difference in demand between (At start t0 and now)
        return demand[1:], demand[0] - latest_demand

    def __compute_demand(self, x, d):
        return ((d[1] - d[0]) * x + (d[0] + d[1]) / 2) / 3600.0 * self.timestep * self.amplification_factor

    # Load demand from DB for each hour
    def update_hourly_demand(self, t, max_hours=4):
        localtime = get_local_datetime(t - 60 * 30)
        current_time = localtime.month, localtime.day, localtime.hour
        if len(self.hourly_demand) == 0 or self.current_time != current_time:
            self.current_time = current_time
            self.hourly_demand = [self.load_demand_profile(t + 60 * (60 * i - 30)) for i in range(max_hours)]

        x = (localtime.minute - 30) / 60.0
        return x        # Fraction of hours

    @staticmethod
    def load_demand_profile(t):
        localtime = get_local_datetime(t)
        dayofweek, hour = localtime.weekday(), localtime.hour
        query = """
          SELECT x, y, demand
          FROM demand_profile
          WHERE dayofweek = {dayofweek} and hour = {hour};
                """.format(dayofweek=dayofweek, hour=hour)
        demand = pd.read_sql(query, engine, index_col=["x", "y"]).demand
        M = np.zeros((MAP_WIDTH, MAP_HEIGHT))
        for (x, y), c in demand.iteritems():
            M[x, y] += c
        return M

    @staticmethod
    def load_OD_matrix(t, alpha=0.1):
        localtime = get_local_datetime(t)
        dayofweek, hour = localtime.weekday(), localtime.hour
        hours_bin = int(hour / DESTINATION_PROFILE_TEMPORAL_AGGREGATION)
        query = """
          SELECT origin_x, origin_y, destination_x, destination_y, demand, trip_time
          FROM od_profile
          WHERE dayofweek = {dayofweek} and hours_bin = {hours_bin};
                """.format(dayofweek=dayofweek, hours_bin=hours_bin)
        df = pd.read_sql(query, engine, index_col=["origin_x", "origin_y", "destination_x", "destination_y"])
        X_size = int(MAP_WIDTH / DESTINATION_PROFILE_SPATIAL_AGGREGATION) + 1
        Y_size = int(MAP_HEIGHT / DESTINATION_PROFILE_SPATIAL_AGGREGATION) + 1
        OD = np.full((X_size, Y_size, X_size, Y_size), alpha)
        TT = np.zeros((X_size, Y_size, X_size, Y_size))
        for od, row in df.iterrows():
            OD[od] += row.demand
            TT[od] = row.trip_time
        for ox in range(X_size):
            for oy in range(Y_size):
                OD[ox, oy] /= OD[ox, oy].sum()
        average_TT = np.zeros((X_size, Y_size))
        for ox in range(X_size):
            for oy in range(Y_size):
               average_TT[ox, oy] = (TT[ox, oy] * OD[ox, oy]).sum()
        # TT = np.tensordot(TT, OD, axes=[(2, 3), (2, 3)])
        return OD, average_TT

    @staticmethod
    def load_latest_demand(t_start, t_end):
        query = """
          SELECT x, y, demand
          FROM demand_latest
          WHERE t > {t_start} and t <= {t_end};
                """.format(t_start = t_start, t_end = t_end)
        demand = pd.read_sql(query, engine, index_col=["x", "y"]).demand
        M = np.zeros((MAP_WIDTH, MAP_HEIGHT))
        for (x, y), c in demand.iteritems():
            M[x, y] += c
        return M

