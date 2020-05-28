from logger import sim_logger
from novelties import status_codes, customer_preferences, vehicle_types
from simulator.settings import FLAGS
from random import randrange

class Customer(object):
    def __init__(self, request):
        self.request = request
        self.status = status_codes.C_CALLING
        self.waiting_time = 0
        self.car_preference = 0
        self.RS_preference = 0
        self.time_preference = 0
        self.max_tolerate_delay = 0
        self.price_threshold = 0
        self.go_to_nxt_timestep = 0
        self.accepted_price = 0
        self.set_preferences()

    def set_preferences(self):
        i = randrange(2)
        if i == 0:
            self.car_preference = customer_preferences.C_any_car
            self.price_threshold = 20
        else:
            self.car_preference = customer_preferences.C_luxury_car
            self.price_threshold = 30
        i = randrange(2)
        if i == 0:
            self.time_preference = customer_preferences.C_not_inHurry
            self.max_tolerate_delay = float(600)
        else:
            self.time_preference = customer_preferences.C_inHurry
            self.max_tolerate_delay = float(120)
        i = randrange(2)
        if i == 0:
            self.RS_preference = customer_preferences.C_not_rideShare
        else:
            self.car_preference = customer_preferences.C_rideShare

    def step(self, timestep):
        if self.status == status_codes.C_CALLING and not self.go_to_nxt_timestep:
            self.disappear()

    def to_string(self):
        s = str(self.request.id) + " " + str(self.status) + " " + str(self.waiting_time)
        return s

    def print_customer(self):
        print("\n Customer Info")
        # print(self.state)
        print(self.request.id)
        print(self.request.origin_lat, self.request.origin_lon)
        print(self.request.destination_lat, self.request.destination_lon, self.request.trip_time)
        print(self.status, self.waiting_time)

    def get_id(self):
        customer_id = self.request.id
        return customer_id

    def get_origin(self):
        # The initial location for the request
        origin = self.request.origin_lat, self.request.origin_lon
        return origin

    def get_destination(self):
        # The location of the destination
        destination = self.request.destination_lat, self.request.destination_lon
        return destination

    def get_trip_duration(self):
        trip_time = self.request.trip_time
        return trip_time

    def get_request(self):
        return self.request

    # Customer utility function goes here
    def accpet_reject_ride(self, initial_price, assigned_vehicle_status, time_till_pickup):
        accept_response = 0
        capacity = assigned_vehicle_status.current_capacity
        threshold = 0
        # if self.RS_preference == customer_preferences.C_not_rideShare and capacity > 0:
        #     return 0
        #
        # if time_till_pickup > self.max_tolerate_delay and self.time_preference == customer_preferences.C_inHurry:
        #     return 0

        # response = randrange(2)
        # utility = 1/initial_price
        # if utility >= 1/self.price_threshold:
        #     accept_response = 1
        if time_till_pickup <= 0:
            time_till_pickup = 3600.0
        utility = (15.0/(capacity+1)) + (4.0*assigned_vehicle_status.type) + (3600.0/time_till_pickup)
        # print("P: ", initial_price, "P - 10: ", initial_price-10)
        if assigned_vehicle_status.type == vehicle_types.hatch_back:
            accept_response = 1
            return accept_response
        elif assigned_vehicle_status.type == vehicle_types.sedan:
            threshold = float(9)
        elif assigned_vehicle_status.type == vehicle_types.SUV:
            threshold = float(13)
        else:
            threshold = float(17)
        if utility > (initial_price - threshold):
            accept_response = 1
            self.accepted_price = initial_price
        else:
            self.accepted_price = 0
        return accept_response

    def wait_for_vehicle(self, waiting_time):
        self.waiting_time = waiting_time
        self.status = status_codes.C_WAITING

    def ride_on(self):
        self.status = status_codes.C_IN_VEHICLE
        self.__log()

    def get_off(self):
        self.status = status_codes.C_ARRIVED
        self.__log()

    def disappear(self):
        self.status = status_codes.C_DISAPPEARED
        self.__log()

    def is_arrived(self):
        return self.status == status_codes.C_ARRIVED

    def is_disappeared(self):
        return self.status == status_codes.C_DISAPPEARED

    def make_payment(self, base):
        if self.accepted_price < base:
            # print(base)
            return base
        else:
            # print(self.accepted_price)
            return self.accepted_price

    def __log(self):
        msg = ','.join(map(str, [self.request.id, self.status, self.waiting_time]))
        sim_logger.log_customer_event(msg)
