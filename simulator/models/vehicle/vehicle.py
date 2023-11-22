from novelties import status_codes
from common import geoutils
from config.settings import MIN_WORKING_TIME, MAX_WORKING_TIME
from .vehicle_state import VehicleState
from .vehicle_behavior import Occupied, Cruising, Idle, Assigned, OffDuty
from logger import sim_logger
from logging import getLogger
import numpy as np
from simulator.services.routing_service import RoutingEngine
from simulator.settings import FLAGS
from simulator.models.customer.customer_repository import CustomerRepository
from common.geoutils import great_circle_distance


class Vehicle(object):
    behavior_models = {
        status_codes.V_IDLE: Idle(),
        status_codes.V_CRUISING: Cruising(),
        status_codes.V_OCCUPIED: Occupied(),
        status_codes.V_ASSIGNED: Assigned(),
        status_codes.V_OFF_DUTY: OffDuty(),
    }

    def __init__(self, vehicle_state):
        if not isinstance(vehicle_state, VehicleState):
            raise ValueError
        self.state = vehicle_state
        # print(self.state.type, self.state.max_capacity)
        self.__behavior = self.behavior_models[vehicle_state.status]
        self.onboard_customers = (
            []
        )  # A vehicle can have a list of customers  #Customers already picked up
        self.accepted_customers = []  # Customers to be picked up
        self.ordered_pickups_dropoffs_ids = (
            []
        )  # Queue of IDs to determine next customer to be picked up or dropped off
        self.pickup_flags = (
            []
        )  # Queue of Flags to determine whether next stop is pickup or dropof
        self.current_plan = []  # Queue of locations determining th current planned trip
        self.current_plan_routes = (
            []
        )  # Queue of routes corresponding to each location on the current trip
        self.nxt_stop = None
        self.__customers_ids = []
        self.__route_plan = []
        self.earnings = 0
        self.working_time = 0
        self.num_trip_customers = 0
        self.epsilon = 5
        self.first_dispatched = 0
        self.pickup_time = 0
        self.tmp_capacity = 0
        self.routing_engine = RoutingEngine.create_engine()
        self.q_action_dict = {}
        self.duration = np.zeros(len(self.behavior_models))  # Duration for each state

    # state changing methods
    def step(self, timestep):
        # print(self.state.id, "Loc: ", self.state.lon, self.state.lat)
        # print(self.state.id, "C: ", self.state.current_capacity)
        self.working_time += timestep
        if self.state.status == status_codes.V_OCCUPIED:
            self.duration[status_codes.V_OCCUPIED] += timestep
        elif self.state.status == status_codes.V_CRUISING:
            self.duration[status_codes.V_CRUISING] += timestep
        # elif self.state.status == status_codes.V_OFF_DUTY:
        #     self.duration[status_codes.V_OFF_DUTY] += timestep
        elif self.state.status == status_codes.V_ASSIGNED:
            self.duration[status_codes.V_ASSIGNED] += timestep
        if self.state.status == status_codes.V_IDLE:
            self.duration[status_codes.V_IDLE] += timestep

        if (
            self.state.status
            == status_codes.V_IDLE | self.state.status
            == status_codes.V_CRUISING
        ):
            self.state.idle_duration += timestep
            # self.state.total_idle += timestep
        else:
            self.state.idle_duration = 0

        try:
            self.__behavior.step(self, timestep)
        except:
            logger = getLogger(__name__)
            logger.error(self.state.to_msg())
            raise

    # Calculate Vehicle's spped, based on route and ETA
    def compute_speed(self, route, triptime):
        lats, lons = zip(*route)
        distance = geoutils.great_circle_distance(
            lats[:-1], lons[:-1], lats[1:], lons[1:]
        )  # Distance in meters
        speed = sum(distance) / triptime
        # print("Dispatch!")
        # self.state.travel_dist += sum(distance)
        return speed

    # Calculate fuel consumption based on travel distance, mileage and gas prices
    def compute_fuel_consumption(self):
        return float(
            (
                self.state.travel_dist
                * (self.state.gas_price / (self.state.mileage * 1000.0))
            )
            / 100.0
        )

    # Calculate profit, after subtracting fuel cost from earnings
    def compute_profit(self):
        cost = self.compute_fuel_consumption() / 100.0
        return self.earnings - cost

    # Perform the dispatch by moving to destination
    def cruise(self, route, triptime):
        assert self.__behavior.available
        speed = self.compute_speed(route, triptime)
        self.reset_plan()
        self.set_route(route, speed)
        self.set_destination(route[-1], triptime)
        self.__change_to_cruising()
        self.__log()

    # Set destination to assigned customer's pickup location,
    def head_for_customer(self, triptime, customer_id, route):
        assert self.__behavior.available
        # self.__reset_plan()
        # self.set_destination(destination, triptime)
        self.state.assigned_customer_id = customer_id
        self.__customers_ids.append(customer_id)

        # if not FLAGS.enable_pooling:
        # print("Head, not pooling!")
        pick_drop = self.pickup_flags[0]
        if pick_drop == 1:
            self.change_to_assigned()
        else:
            self.change_to_occupied()
            # self.state.travel_dist += distance
        # else:
        # print(self.get_id(), "Heading: ", pick_drop, self.state.status, self.current_plan,
        #       self.ordered_pickups_dropoffs_ids, self.pickup_flags)
        speed = self.compute_speed(route, triptime)
        self.set_route(route, speed)
        self.set_destination(route[-1], triptime)

        self.__log()

    def take_rest(self, duration):
        assert self.__behavior.available
        self.reset_plan()
        self.state.idle_duration = 0  # Resetting the idle time
        self.set_destination(self.get_location(), duration)
        self.__change_to_off_duty()
        self.__log()

    # Upon reaching pickup location, adding the customer to the on-board customers and heading to next customer.
    def pickup(self, customer):
        # print("At Pickup!", self.get_location(), " -> ", customer.get_origin())
        # print(self.get_id(), "Pickup: ", self.current_plan, self.ordered_pickups_dropoffs_ids)
        customer.ride_on()
        # customer_id = customer.get_id()
        # self.__reset_plan() # For now we don't consider routes of occupied trip
        self.onboard_customers.append(customer)
        self.num_trip_customers += 1
        self.state.current_capacity += 1
        self.tmp_capacity += 1

        if len(self.current_plan) == 0:
            print("pickup -> NEVER")
            self.change_to_idle()
            self.state.price_per_travel_m /= self.num_trip_customers
            self.num_trip_customers = 0
            self.accepted_customers = []
            self.onboard_customers = []
            self.reset_plan()

        else:
            self.nxt_stop = self.current_plan.pop(0)
            pick_drop = self.pickup_flags[0]
            # routes = self.routing_engine.route([(self.get_location(), self.nxt_stop)])
            route, triptime = self.current_plan_routes.pop(0)
            # print("Pickup: ", triptime, route)
            # print(vehicle.get_location(), vehicle.current_plan[0])
            # If already there at nxt stop
            if triptime == 0.0:
                self.pickup_flags.pop(0)
                id = self.ordered_pickups_dropoffs_ids.pop(0)
                self.state.lat, self.state.lon = self.nxt_stop
                if pick_drop == 1:
                    if self.get_location() != CustomerRepository.get(id).get_origin():
                        self.state.lat, self.state.lon = CustomerRepository.get(
                            id
                        ).get_origin()
                    self.pickup(CustomerRepository.get(id))
                else:
                    if (
                        self.get_location()
                        != CustomerRepository.get(id).get_destination()
                    ):
                        self.state.lat, self.state.lon = CustomerRepository.get(
                            id
                        ).get_destination()
                    self.dropoff(CustomerRepository.get(id))
            else:
                speed = self.compute_speed(route, triptime)
                self.set_route(route, speed)
                self.set_destination(route[-1], triptime)

            # self.state.assigned_customer_id = customer_id
            # triptime = customer.get_trip_duration()
            # self.__set_destination(customer.get_destination(), triptime)
            self.__set_pickup_time(triptime)
            self.change_to_occupied()
        # self.state.current_capacity += 1
        self.__log()

    # Upon reaching customer's destination, remove customer from onboard and collecting fare
    def dropoff(self, customer):
        self.onboard_customers.remove(customer)
        customer.get_off()

        self.state.current_capacity -= 1
        self.tmp_capacity -= 1

        if (self.state.current_capacity == 0) | (len(self.current_plan) == 0):
            # print(self.get_id(), "IDLE!")
            self.change_to_idle()
            self.state.status = status_codes.V_IDLE
            self.state.price_per_travel_m /= self.num_trip_customers
            self.num_trip_customers = 0
            self.accepted_customers = []
            self.onboard_customers = []
            # latest_cust_id = self.state.assigned_customer_id.pop(0)
            self.reset_plan()

        else:
            self.nxt_stop = self.current_plan.pop(0)
            pick_drop = self.pickup_flags[0]
            # routes = self.routing_engine.route([(self.get_location(), self.nxt_stop)])
            route, triptime = self.current_plan_routes.pop(0)
            # print("Dropoff: ", triptime, route)
            # print(vehicle.get_location(), vehicle.current_plan[0])
            if triptime == 0.0:
                self.pickup_flags.pop(0)
                id = self.ordered_pickups_dropoffs_ids.pop(0)
                # speed = self.compute_speed(route, triptime)
                # self.set_route(route, speed)
                # self.set_destination(route[-1], triptime)
                self.state.lat, self.state.lon = self.nxt_stop
                if pick_drop == 1:
                    if self.get_location() != CustomerRepository.get(id).get_origin():
                        self.state.lat, self.state.lon = CustomerRepository.get(
                            id
                        ).get_origin()
                    self.pickup(CustomerRepository.get(id))
                else:
                    if (
                        self.get_location()
                        != CustomerRepository.get(id).get_destination()
                    ):
                        self.state.lat, self.state.lon = CustomerRepository.get(
                            id
                        ).get_destination()
                    self.dropoff(CustomerRepository.get(id))
            else:
                # print("Dropoff: ", len(self.current_plan), len(self.ordered_pickups_dropoffs_ids))
                speed = self.compute_speed(route, triptime)
                self.set_route(route, speed)
                self.set_destination(route[-1], triptime)

        self.__log()
        # return customer

    # Switch to parking, idle state and reset plan
    def park(self):
        self.reset_plan()
        self.change_to_idle()
        self.__log()

    def update_location(self, location, route):
        self.state.lat, self.state.lon = location
        self.__route_plan = route

    # subtract time travelled from ETA to updatime time to destination
    def update_time_to_destination(self, timestep):
        dt = min(timestep, self.state.time_to_destination)
        self.duration[self.state.status] += dt
        self.state.time_to_destination -= dt
        if self.state.time_to_destination <= 0:
            self.state.time_to_destination = 0
            self.state.lat = self.state.destination_lat
            self.state.lon = self.state.destination_lon
            return True
        else:
            return False

    # some getter methods
    def get_id(self):
        vehicle_id = self.state.id
        return vehicle_id

    def get_customers_ids(self):
        return self.__customers_ids

    def get_location(self):
        location = self.state.lat, self.state.lon
        return location

    def get_destination(self):
        destination = self.state.destination_lat, self.state.destination_lon
        return destination

    def get_speed(self):
        speed = self.state.speed
        return speed

    def get_price_rates(self):
        return [self.state.price_per_travel_m, self.state.price_per_wait_min]

    def reachedCapacity(self):
        if self.state.current_capacity == self.state.max_capacity:
            return True
        else:
            return False

    def get_assigned_customer_id(self):
        customer_id = self.state.assigned_customer_id
        return customer_id

    def to_string(self):
        s = (
            str(getattr(self.state, "id"))
            + " Capacity: "
            + str(self.state.current_capacity)
        )
        return s

    def print_vehicle(self):
        print("\n Vehicle Info")
        for attr in self.state.__slots__:
            print(attr, " ", getattr(self.state, attr))

        print("IDS::", self.__customers_ids)
        # print(self.state)
        print(self.__behavior)
        for cus in self.onboard_customers:
            cus.print_customer()
        # print(self.__route_plan)
        print("earnings", self.earnings)
        print("working_time", self.working_time)
        print("current_capacity", self.state.current_capacity)
        # print(self.duration)

    def get_route(self):
        return self.__route_plan[:]

    def get_total_dist(self):
        return self.state.travel_dist

    def get_idle_duration(self):
        dur = (
            self.working_time
            - self.duration[status_codes.V_OCCUPIED]
            - self.duration[status_codes.V_ASSIGNED]
        )
        # print(self.duration)
        return dur

    # def get_pickup_time(self):
    #     return self.pickup_time

    def get_state(self):
        state = []
        for attr in self.state.__slots__:
            state.append(getattr(self.state, attr))
        return state

    def get_score(self):
        score = [self.working_time, self.earnings] + self.duration.tolist()
        return score

    def get_num_cust(self):
        return self.state.current_capacity

    def get_vehicle(self, id):
        if id == self.state.id:
            return self

    def exit_market(self):
        if self.__behavior.available:
            if self.state.idle_duration == 0:
                return self.working_time > MIN_WORKING_TIME
            else:
                return self.working_time > MAX_WORKING_TIME
        else:
            return False

    def reset_plan(self):
        self.state.reset_plan()
        self.__route_plan = []

    def set_route(self, route, speed):
        # assert self.get_location() == route[0]
        self.__route_plan = route
        self.state.speed = speed

    def set_destination(self, destination, triptime):
        self.state.destination_lat, self.state.destination_lon = destination
        self.state.time_to_destination = triptime

    def __set_pickup_time(self, triptime):
        self.pickup_time = triptime

    def change_to_idle(self):
        self.__change_behavior_model(status_codes.V_IDLE)

    def __change_to_cruising(self):
        self.__change_behavior_model(status_codes.V_CRUISING)

    def change_to_assigned(self):
        self.__change_behavior_model(status_codes.V_ASSIGNED)

    def change_to_occupied(self):
        self.__change_behavior_model(status_codes.V_OCCUPIED)

    def __change_to_off_duty(self):
        self.__change_behavior_model(status_codes.V_OFF_DUTY)

    def __change_behavior_model(self, status):
        self.__behavior = self.behavior_models[status]
        self.state.status = status

    def __log(self):
        # if FLAGS.log_vehicle:
        sim_logger.log_vehicle_event(self.state.to_msg())
