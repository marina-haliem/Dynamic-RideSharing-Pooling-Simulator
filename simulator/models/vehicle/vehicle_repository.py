from .vehicle import Vehicle
from .vehicle_state import VehicleState
import pandas as pd

class VehicleRepository(object):
    vehicles = {}

    @classmethod
    def init(cls):
        cls.vehicles = {}

    @classmethod
    def populate(cls, vehicle_id, location, type):
        # populate vehicles over the city maps.
        state = VehicleState(vehicle_id, location, type)
        cls.vehicles[vehicle_id] = Vehicle(state)

    @classmethod
    def get_all(cls):
        # returns a list of all vehicles in repository.
        return list(cls.vehicles.values())

    @classmethod
    def get(cls, vehicle_id):
        return cls.vehicles.get(vehicle_id, None)

    @classmethod
    def get_states(cls):
        # get all vehicles' state attributes from repository.
        states = [vehicle.get_state() for vehicle in cls.get_all()]
        cols = VehicleState.__slots__[:]
        df = pd.DataFrame.from_records(states, columns=cols).set_index("id")    # Creating DF with all attributes in Vehicle State
        df["earnings"] = [vehicle.earnings for vehicle in cls.get_all()]
        df["pickup_time"] = [vehicle.pickup_time for vehicle in cls.get_all()]
        df["cost"] = [vehicle.compute_fuel_consumption() for vehicle in cls.get_all()]
        df["total_idle"] = [vehicle.get_idle_duration() for vehicle in cls.get_all()]
        # df["agent_type"] = [vehicle.get_idle_duration() for vehicle in cls.get_all()]
        # print(df.columns.names)
        return df


    @classmethod
    def delete(cls, vehicle_id):
        # delete when vehicle leaves the market.
        cls.vehicles.pop(vehicle_id)
