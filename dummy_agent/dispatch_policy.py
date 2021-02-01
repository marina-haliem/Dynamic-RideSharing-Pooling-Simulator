from novelties import status_codes
# from .services.demand_prediction_service import DemandPredictionService
from config.settings import TIMESTEP, MIN_DISPATCH_CYCLE, MAX_DISPATCH_CYCLE
import numpy as np

class DispatchPolicy(object):
    def __init__(self):
        # self.demand_predictor = DemandPredictionService()
        self.updated_at = {}

    def dispatch(self, current_time, vehicles):
        self.update_state(current_time, vehicles)
        tbd_vehicles = self.get_tbd_vehicles(vehicles, current_time)
        # print("Len: ", len(tbd_vehicles))
        if len(tbd_vehicles) == 0:
            return []

        commands = self.get_dispatch_decisions(tbd_vehicles)
        self.record_dispatch(tbd_vehicles.index, current_time)
        return commands

    def update_state(self, current_time, vehicles):
        pass

    def get_dispatch_decisions(self, tbd_vehicles):
        return []

    def get_tbd_vehicles(self, vehicles, current_time):
        idle_vehicles = vehicles[vehicles.status == status_codes.V_IDLE]
        cruising_vehicles = vehicles[vehicles.status == status_codes.V_CRUISING]
        # print("I:", len(idle_vehicles))
        # print("C:", len(cruising_vehicles))
        # Retrieve idle vehicles to be dispatched -> those that exceeded the min dispatch cycle being idle
        tbd_idle_vehicles = idle_vehicles.loc[[
            vehicle_id for vehicle_id in idle_vehicles.index
            if current_time - self.updated_at.get(vehicle_id, 0) >= MIN_DISPATCH_CYCLE
        ]]
        # Retrieve cruising vehicles to be dispatched -> those that exceeded the max dispatch cycle
        tbd_cruising_vehicles = cruising_vehicles.loc[[
            vehicle_id for vehicle_id in cruising_vehicles.index
            if current_time - self.updated_at.get(vehicle_id, 0) >= MAX_DISPATCH_CYCLE
        ]]

        tbd_vehicles = tbd_idle_vehicles.append(tbd_cruising_vehicles)
        # Max num of vehicles that can be dispatched in one dispatch cycle
        # One dispatch cycle = MIN_DISPATCH_CYCLE * TIMESTEP
        max_n = int(len(vehicles) / MIN_DISPATCH_CYCLE * TIMESTEP)
        if len(tbd_vehicles) > max_n:
            # All permutations of the list of vehicles to be dispatched
            p = np.random.permutation(len(tbd_vehicles))
            # Stripping the list to max_n
            tbd_vehicles = tbd_vehicles.iloc[p[:max_n]]
        return tbd_vehicles

    # Store time of dispatch for each vehicle
    def record_dispatch(self, vehicle_ids, current_time):
        for vehicle_id in vehicle_ids:
            self.updated_at[vehicle_id] = current_time

    # Creating Dispatch dictionary associated with each vehicle ID, it could be decided for that vehicle to be
    # Offduty, or be assigned a destination to head to, or have a cache key
    def create_dispatch_dict(self, vehicle_id, destination=None, offduty=False, cache_key=None):
        dispatch_dict = {}
        dispatch_dict["vehicle_id"] = vehicle_id
        if offduty:
            dispatch_dict["offduty"] = True
        elif cache_key is not None:
            dispatch_dict["cache"] = cache_key
        else:
            dispatch_dict["destination"] = destination
        return dispatch_dict