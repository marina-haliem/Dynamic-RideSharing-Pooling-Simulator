from novelties import status_codes
from novelties import vehicle_types, agent_codes
from random import randrange

class VehicleState(object):
    # State Vector for the vehicle
    __slots__ = [
        'id', 'lat', 'lon', 'speed', 'status', 'destination_lat', 'destination_lon', 'type', 'travel_dist', 'price_per_travel_m', 'price_per_wait_min', 'gas_price',
        'assigned_customer_id', 'time_to_destination', 'idle_duration', 'current_capacity', 'max_capacity',
        'driver_base_per_trip', 'mileage', 'agent_type', 'accept_new_request']


    def __init__(self, id, location, agent_type):
        self.id = id
        self.lat, self.lon = location
        self.speed = 0
        # self.status = vehicle_status_codes.IDLE
        self.status = status_codes.V_IDLE
        self.destination_lat, self.destination_lon = None, None
        self.assigned_customer_id = None
        self.time_to_destination = 0
        self.idle_duration = 0
        # self.total_idle = 0
        self.current_capacity = 0
        self.travel_dist = 0
        self.driver_base_per_trip = 0
        self.price_per_travel_m = 0
        self.price_per_wait_min = 0
        self.gas_price = 235        # In cents
        self.accept_new_request = False
        self.type = self.selectVehicleType()
        self.max_capacity = self.setCapacity()
        self.mileage = self.set_mileage()
        self.set_price_rates()
        self.agent_type = agent_type

    def selectVehicleType(self):
        r = randrange(4)
        if r == 0:
            return vehicle_types.hatch_back
        elif r == 1:
            return vehicle_types.sedan
        elif r == 2:
            return vehicle_types.luxury
        elif r == 3:
            return vehicle_types.SUV
        return r

    def setCapacity(self):
        if self.type == vehicle_types.hatch_back:
            return 3
        if self.type == vehicle_types.sedan:
            return 4
        if self.type == vehicle_types.luxury:
            return 4
        if self.type == vehicle_types.SUV:
            return 5

    def set_mileage(self):
        if self.type == vehicle_types.sedan:
            self.mileage = float(30)
            return self.mileage
        if self.type == vehicle_types.hatch_back:
            self.mileage = float(35)
            return self.mileage
        if self.type == vehicle_types.luxury:
            self.mileage = float(25)
            return self.mileage
        if self.type == vehicle_types.SUV:
            self.mileage = float(15)
            return self.mileage

    def set_price_rates(self):
        if self.type == vehicle_types.sedan:
            self.price_per_travel_m = float(375) / 1000.0  # Per meter
            self.price_per_wait_min = float(0.05 / 3600)  # Per hour
            self.driver_base_per_trip = float(1000)
        if self.type == vehicle_types.hatch_back:
            self.price_per_travel_m = float(350) / 1000.0
            self.price_per_wait_min = float(0.05 / 3600)
            self.driver_base_per_trip = float(500)
        if self.type == vehicle_types.luxury:
            self.price_per_travel_m = float(425) / 1000.0
            self.price_per_wait_min = float(0.05 / 3600)
            self.driver_base_per_trip = float(1600)
        if self.type == vehicle_types.SUV:
            self.price_per_travel_m = float(400) / 1000.0
            self.price_per_wait_min = float(0.05 / 3600)
            self.driver_base_per_trip = float(1500)

    # When trip is over or cancelled
    def reset_plan(self):
        self.destination_lat, self.destination_lon = None, None
        self.speed = 0
        self.assigned_customer_id = None
        self.time_to_destination = 0


    def to_msg(self):
        state = [str(getattr(self, name)).format(":.2f") for name in self.__slots__]
        return ','.join(state)

    # def