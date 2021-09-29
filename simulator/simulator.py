from collections import defaultdict

import numpy as np

from common import geoutils
from simulator.models.vehicle.vehicle_repository import VehicleRepository
from simulator.models.customer.customer_repository import CustomerRepository
from simulator.services.demand_generation_service import DemandGenerator
from simulator.services.routing_service import RoutingEngine
from common.time_utils import get_local_datetime
from config.settings import OFF_DURATION, PICKUP_DURATION
from simulator.settings import FLAGS
from logger import sim_logger
from logging import getLogger
from novelties import agent_codes, status_codes
# from novelties.pricing.price_calculator import recalculate_price_per_customer
from random import randrange
from common.geoutils import great_circle_distance
from novelties.pricing.price_calculator import calculate_price
import itertools

class Simulator(object):
    def __init__(self, start_time, timestep):
        self.reset(start_time, timestep)
        sim_logger.setup_logging(self)
        self.logger = getLogger(__name__)
        self.demand_generator = DemandGenerator()
        self.routing_engine = RoutingEngine.create_engine()
        self.route_cache = {}
        self.current_dummyV = 0
        self.current_dqnV = 0

    def reset(self, start_time=None, timestep=None):
        if start_time is not None:
            self.__t = start_time
        if timestep is not None:
            self.__dt = timestep
        VehicleRepository.init()
        CustomerRepository.init()

    def populate_vehicle(self, vehicle_id, location):
        type = 0
        r = randrange(2)
        if r == 0 and self.current_dummyV < FLAGS.dummy_vehicles:
            type = agent_codes.dummy_agent
            self.current_dummyV += 1

        # If r = 1 or num of dummy agent satisfied
        elif self.current_dqnV < FLAGS.dqn_vehicles:
            type = agent_codes.dqn_agent
            self.current_dqnV += 1

        else:
            type = agent_codes.dummy_agent
            self.current_dummyV += 1

        VehicleRepository.populate(vehicle_id, location, type)

    def step(self):
        for customer in CustomerRepository.get_all():
            customer.step(self.__dt)
            if customer.is_arrived() or customer.is_disappeared():
                CustomerRepository.delete(customer.get_id())

        for vehicle in VehicleRepository.get_all():
            vehicle.step(self.__dt)
            # vehicle.print_vehicle()
            if vehicle.exit_market():
                score = ','.join(map(str, [self.get_current_time(), vehicle.get_id(), vehicle.get_total_dist(),
                                           vehicle.compute_profit()] + vehicle.get_score()))
                if vehicle.state.agent_type == agent_codes.dqn_agent:
                    self.current_dqnV -= 1
                else:
                    self.current_dummyV -= 1
                sim_logger.log_score(score)
                VehicleRepository.delete(vehicle.get_id())

        self.__populate_new_customers()
        self.__update_time()
        if self.__t % 3600 == 0:
            # print("Elapsed : {}".format(get_local_datetime(self.__t)))
            self.logger.info("Elapsed : {}".format(get_local_datetime(self.__t)))

    def match_vehicles(self, commands, dqn_agent, dummy_agent):
        # print("M: ", commands)
        vehicle_list = []
        rejected_requests = []
        accepted_commands = []
        num_accepted = 0
        # reject_count = 0
        vehicle_accepted_cust = defaultdict(list)
        # od_accepted_pairs = []
        # Comamnd is a dictionary created in dummy_agent
        # print("########################################################")
        for command in commands:
            rejected_flag = 0
            # print(command["vehicle_id"], command["customer_id"])
            vehicle = VehicleRepository.get(command["vehicle_id"])
            vid = command["vehicle_id"]
            # print("V_Loc: ", vehicle.get_location())
            # vehicle.state.status = status_codes.V_ASSIGNED
            if vehicle is None:
                self.logger.warning("Invalid Vehicle id")
                continue
            # print("Vid: ", vid, "Plan: ", vehicle.current_plan)
            # vehicle_cust_price_time = dict()

            if (vehicle.state.status == status_codes.V_OCCUPIED) & vehicle.state.accept_new_request:
                # print(vid, "Update", vehicle.current_plan, vehicle.get_destination())
                if (len(vehicle.current_plan) == 0) & (vehicle.state.destination_lat is not None):
                    # print(vid, "Dest: ", vehicle.get_destination())
                    vehicle.current_plan = [vehicle.get_destination()]
                    vehicle.current_plan_routes = [(vehicle.get_route(), vehicle.state.time_to_destination)]

                elif (len(vehicle.current_plan) == 0) & (vehicle.state.destination_lat is None):
                    vehicle.change_to_idle()
                    vehicle.reset_plan()

                elif (len(vehicle.current_plan) != 0) & (
                        vehicle.get_destination() != vehicle.current_plan[0]):
                    # print(vid, ": ", vehicle.get_destination(), vehicle.current_plan[0])
                    if vehicle.state.destination_lat is not None:
                        plan = [vehicle.get_destination()]
                        plan.extend(vehicle.current_plan)
                        vehicle.current_plan = np.copy(plan).tolist()

                        if len(vehicle.get_route()) == 0:
                            # print(vid, "Empty Route!!", vehicle.get_destination())
                            # print(vehicle.current_plan)
                            # print(vehicle.pickup_flags)
                            # print(vehicle.ordered_pickups_dropoffs_ids)
                            # print(vehicle.current_plan_routes)

                            od_routes = self.routing_engine.route_time([(vehicle.get_location(),
                                                                     vehicle.get_destination())])
                            routes = [od_routes[0]]
                        else:

                            routes = [[vehicle.get_route(), vehicle.state.time_to_destination]]

                        routes.extend(vehicle.current_plan_routes)
                        vehicle.current_plan_routes = np.copy(routes).tolist()

                        # if len(vehicle.get_route()) == 0:
                        #     print("R: ", vehicle.current_plan_routes)

                if len(vehicle.current_plan) != len(vehicle.current_plan_routes) != len(\
                        vehicle.ordered_pickups_dropoffs_ids) != len(vehicle.pickup_flags):
                    print("ERROR!")

            prev_cost = 0
            # For each vehicle
            # Need to calculate route (and order customer list) before heading to customer
            for index in range(len(command["customer_id"])):
                customer = CustomerRepository.get(command["customer_id"][index])
                if customer is None:
                    self.logger.warning("Invalid Customer id")
                    continue

                prev_plan = np.copy(vehicle.current_plan)
                prev_flags = np.copy(vehicle.pickup_flags)
                prev_ids = np.copy(vehicle.ordered_pickups_dropoffs_ids)
                prev_routes = np.copy(vehicle.current_plan_routes)

                vehicle_accepted_cust[vid].append(command["customer_id"][index])

                insertion_cost, waiting_time = self.generate_plan(vehicle, vehicle_accepted_cust[vid], customer)

                if len(vehicle_accepted_cust[vid]) > 1:
                    # print("prev: ", prev_cost, insertion_cost)
                    insertion_cost = abs(insertion_cost - prev_cost)

                if insertion_cost == float(0):
                    insertion_cost = command["init_price"][index]
                # print("L: ", len(vehicle_accepted_cust[vid]))
                # print("C: ", insertion_cost)

                [travel_price, wait_price] = vehicle.get_price_rates()

                init_price = calculate_price(insertion_cost, waiting_time, vehicle.state.mileage, travel_price,
                                                wait_price, vehicle.state.gas_price,
                                                vehicle.state.driver_base_per_trip)

                # print("P: ", init_price, command["init_price"][index])
                command["duration"][index] = waiting_time

                command["init_price"][index] = init_price

                # print("A: ", command)

                if FLAGS.enable_pricing:
                    # For DQN
                    if vehicle.state.agent_type == agent_codes.dummy_agent:
                        price_response = dummy_agent.get_price_decision(vehicle, init_price, customer.get_request())

                    elif vehicle.state.agent_type == agent_codes.dqn_agent:
                        price_response = dqn_agent.get_price_decision(vehicle, init_price, customer.get_request())
                    # price_response = initial_price

                    # print("Diff: ", (price_response-initial_price))

                    # Now, customer needs to calculate reward and accept or reject
                    if customer.accpet_reject_ride(price_response, vehicle.state, waiting_time):   # If customer accepts
                        num_accepted += 1
                        # if not FLAGS.enable_pooling:
                        #     vehicle.head_for_customer(customer.get_origin(), waiting_time, customer.get_id(), command["distance"][index])
                        vehicle.accepted_customers.append(command["customer_id"][index])
                        # vehicle.accepted_customers.append([customer, triptime, price_response, command["distance"][index]])

                        customer.wait_for_vehicle(waiting_time)
                        prev_cost = insertion_cost

                        # vehicle_accepted_cust[vid].append(command["customer_id"][index])
                        # vehicle.state.current_capacity += 1
                        accepted_commands.append(command)
                        # print("Accepted, cust: ", customer.get_id(), " ", vehicle.current_plan)


                    else:
                        # rejected_flag = 1
                        # reject_count += 1
                        customer.go_to_nxt_timestep = 1
                        rejected_requests.append(customer.get_request())
                        # if len(prev_plan) == 0:
                        #     print("Should be Empty!")
                        vehicle.current_plan = prev_plan.tolist()
                        vehicle.pickup_flags = prev_flags.tolist()
                        vehicle.ordered_pickups_dropoffs_ids = prev_ids.tolist()
                        vehicle.current_plan_routes = prev_routes.tolist()
                        # print("B: ", vehicle.accepted_customers, vehicle_accepted_cust[vid])
                        # print("Reject: ", vid, " ", customer.get_id())
                        vehicle_accepted_cust[vid].pop()
                        # vehicle.accepted_customers.pop(0)
                        # print("A: ", vehicle_accepted_cust[vid], len(vehicle_accepted_cust[vid]), vehicle.accepted_customers)
                        # print("Rejected, cust: ", customer.get_id(), " ", vehicle.current_plan)

                else:
                    customer.accepted_price = init_price
                    # if not FLAGS.enable_pooling:
                    #     vehicle.head_for_customer(customer.get_origin(), triptime, customer.get_id(), command["distance"][index])
                    vehicle.accepted_customers.append(command["customer_id"][index])
                    # vehicle.accepted_customers.append([customer, triptime, price_response, command["distance"][index]])
                    customer.wait_for_vehicle(waiting_time)
                    prev_cost = insertion_cost
                    accepted_commands = commands
                    # od_accepted_pairs = pairs
                    # vehicle.state.status = status_codes.V_ASSIGNED

            if FLAGS.enable_pooling:
                # print("vid: ", vehicle.get_id(), "Accepted: ", len(vehicle_accepted_cust[vid]))
                vehicle.tmp_capacity = vehicle.state.current_capacity
                # if vid not in vehicle_list and len(vehicle_accepted_cust[vid]) != 0:
                #     vehicle_list.append(vid)

                if len(vehicle.current_plan) == 0:
                    # print(vid, "EMPTYYYYYY!")
                    continue

                else:
                    route, triptime = vehicle.current_plan_routes.pop(0)
                    vehicle.nxt_stop = vehicle.current_plan.pop(0)
                    if len(route) == 0:
                        # print("B: ", triptime, len(vehicle.current_plan), len(vehicle.ordered_pickups_dropoffs_ids),
                        #       len(vehicle.current_plan_routes))
                        r2 = self.routing_engine.route_time([(vehicle.get_location(), vehicle.nxt_stop)])
                        route, triptime = r2[0]
                        # print("Updated: ", triptime, len(route))
                    # print("Loc: ", vehicle.get_location(), "Nxt: ", vehicle.nxt_stop)
                    cust_id = vehicle.ordered_pickups_dropoffs_ids[0]
                    if triptime == 0.0:
                        # vehicle.current_plan.pop(0)
                        pick_drop = vehicle.pickup_flags.pop(0)
                        cust_id = vehicle.ordered_pickups_dropoffs_ids.pop(0)
                        # print("routes: ", routes)

                        vehicle.state.assigned_customer_id = cust_id
                        if pick_drop == 1:
                            vehicle.state.lat, vehicle.state.lon = CustomerRepository.get(cust_id).get_origin()
                            vehicle.pickup(CustomerRepository.get(cust_id))
                        else:
                            vehicle.state.lat, vehicle.state.lon = CustomerRepository.get(cust_id).get_destination()
                            vehicle.dropoff(CustomerRepository.get(cust_id))
                    else:
                        vehicle.head_for_customer(triptime, cust_id, route)
                        # vehicle.nxt_stop = vehicle.current_plan[0]
                        vehicle.change_to_assigned()

        return rejected_requests, accepted_commands, num_accepted


    def generate_plan(self, vehicle, cust_list, new_customer):
        # print("In Plan: ", vehicle.get_id())
        time_till_pickup = 0
        insertion_cost = 0
        # distance_till_dropoff = 0
        if len(cust_list) == 1 & len(vehicle.current_plan) == 0:
            vehicle.current_plan.append(new_customer.get_origin())
            vehicle.current_plan.append(new_customer.get_destination())
            vehicle.ordered_pickups_dropoffs_ids.append(new_customer.get_id())
            vehicle.ordered_pickups_dropoffs_ids.append(new_customer.get_id())
            vehicle.pickup_flags.append(1)
            vehicle.pickup_flags.append(0)

            routes_till_pickup = [(vehicle.get_location(), vehicle.current_plan[0])]
            routes = self.routing_engine.route_time(routes_till_pickup)

            for (route, time) in routes:
                time_till_pickup += time

            od_pairs = [(vehicle.get_location(), vehicle.current_plan[0]),
                        (vehicle.current_plan[0], vehicle.current_plan[1])]
            routes = self.routing_engine.route_time(od_pairs)
            for (route, time) in routes:
                vehicle.current_plan_routes.append([route, time])
                lats, lons = zip(*route)
                distance = geoutils.great_circle_distance(lats[:-1], lons[:-1], lats[1:],
                                                          lons[1:])  # Distance in meters
                insertion_cost += sum(distance)

        else:
            new_list = [new_customer.get_origin(), new_customer.get_destination()]
            # print(new_list)
            final_pickup_flags = []
            final_plan = []
            final_pickups_dropoffs_ids = []
            final_routes = []

            min_distance = np.inf
            min_time = np.inf
            pickup_index = 0
            # all_options = vehicle.current_plan + new_list
            # print(all_options)
            # perm = list(itertools.permutations(all_options))
            # print(perm)
            # print(len(set(perm)))
            # print("Insert Pickup")
            for pos in range(len(vehicle.current_plan)+1):
                # print(vehicle.current_plan)
                new_plan = vehicle.current_plan[:pos]
                new_pickup_flags = vehicle.pickup_flags[:pos]
                new_pickups_dropoffs_ids = vehicle.ordered_pickups_dropoffs_ids[:pos]
                new_plan.append(new_list[0])
                new_pickup_flags.append(1)
                new_pickups_dropoffs_ids.append(new_customer.get_id())
                new_plan.extend(vehicle.current_plan[pos:])
                new_pickup_flags.extend(vehicle.pickup_flags[pos:])
                new_pickups_dropoffs_ids.extend(vehicle.ordered_pickups_dropoffs_ids[pos:])

                # print("List: ", new_plan, new_pickup_flags, new_pickups_dropoffs_ids)
                od_pairs = [(vehicle.get_location(), new_plan[0])]
                od_pairs.extend([(new_plan[x], new_plan[x + 1]) for x in range(len(new_plan) - 1)])
                # print(od_pairs)
                total_time = 0
                total_dist = 0
                potential_routes_time = self.routing_engine.route_time(od_pairs)
                new_routes = []
                index = 0
                wait_time = 0
                # pickup_distance = 0
                for (route, time) in potential_routes_time:
                    total_time += time
                    lats, lons = zip(*route)
                    distance = geoutils.great_circle_distance(lats[:-1], lons[:-1], lats[1:],lons[1:])  # Distance in meters
                    total_dist += sum(distance)
                    new_routes.append([route, time])
                    if index < pos+1:
                        wait_time += time
                        # pickup_distance += sum(distance)
                    index += 1
                # print("Pos: ", pos, "Wait: ", wait_time)
                # print("T: ", total_time)
                # print("D: ", total_dist)

                if (total_time < min_time) | (total_dist < min_distance):
                    min_time = total_time
                    min_distance = total_dist
                    insertion_cost = total_dist
                    final_pickup_flags = new_pickup_flags
                    final_plan = new_plan
                    final_pickups_dropoffs_ids = new_pickups_dropoffs_ids
                    pickup_index = pos
                    final_routes = new_routes
                    time_till_pickup = wait_time
                    # distance_till_pickup = pickup_distance

                # print("Min: ", min_time, min_distance)
            vehicle.current_plan = final_plan
            vehicle.pickup_flags = final_pickup_flags
            vehicle.ordered_pickups_dropoffs_ids = final_pickups_dropoffs_ids
            vehicle.current_plan_routes = final_routes

            final_pickup_flags = []
            final_plan = []
            final_pickups_dropoffs_ids = []
            final_routes = []

            min_distance = np.inf
            min_time = np.inf

            # print("Insert Drop-off! ", pickup_index, time_till_pickup, vehicle.current_plan)

            for pos in range(len(vehicle.current_plan)+1):
                if pos <= pickup_index:
                    continue
                # print(vehicle.current_plan)
                new_plan = vehicle.current_plan[:pos]
                new_pickup_flags = vehicle.pickup_flags[:pos]
                new_pickups_dropoffs_ids = vehicle.ordered_pickups_dropoffs_ids[:pos]
                new_plan.append(new_list[1])
                new_pickup_flags.append(0)
                new_pickups_dropoffs_ids.append(new_customer.get_id())
                new_plan.extend(vehicle.current_plan[pos:])
                new_pickup_flags.extend(vehicle.pickup_flags[pos:])
                new_pickups_dropoffs_ids.extend(vehicle.ordered_pickups_dropoffs_ids[pos:])

                # print("List: ", new_plan, new_pickup_flags, new_pickups_dropoffs_ids)
                od_pairs = [(vehicle.get_location(), new_plan[0])]
                od_pairs.extend([(new_plan[x], new_plan[x+1]) for x in range(len(new_plan)-1)])
                total_time = 0
                total_dist = 0
                potential_routes_time = self.routing_engine.route_time(od_pairs)
                # print(len(new_plan)+1, len(od_pairs), len(potential_routes_time))
                new_routes = []
                # counter = 0
                # dropoff_distance = 0
                for (route, time) in potential_routes_time:
                    total_time += time
                    lats, lons = zip(*route)
                    distance = geoutils.great_circle_distance(lats[:-1], lons[:-1], lats[1:],lons[1:])  # Distance in meters
                    total_dist += sum(distance)
                    new_routes.append([route, time])
                    # if pickup_index < counter < pos+1:
                    #     dropoff_distance += sum(distance)
                    # counter += 1
                # print("T: ", total_time)
                # print("D: ", total_dist)

                if (total_time < min_time) | (total_dist < min_distance):
                    min_time = total_time
                    min_distance = total_dist
                    insertion_cost = total_dist
                    final_pickup_flags = new_pickup_flags
                    final_plan = new_plan
                    final_pickups_dropoffs_ids = new_pickups_dropoffs_ids
                    final_routes = new_routes
                    # distance_till_dropoff = dropoff_distance

                # print("Min: ", min_time, min_distance)
            vehicle.current_plan = np.copy(final_plan).tolist()
            vehicle.pickup_flags = np.copy(final_pickup_flags).tolist()
            vehicle.ordered_pickups_dropoffs_ids = np.copy(final_pickups_dropoffs_ids).tolist()
            vehicle.current_plan_routes = np.copy(final_routes).tolist()
            # print("Generated Plan: ", vehicle.current_plan)
            # dist_time = []
            # for (route, time) in vehicle.current_plan_routes:
            #     lats, lons = zip(*route)
            #     distance = geoutils.great_circle_distance(lats[:-1], lons[:-1], lats[1:],lons[1:])  # Distance in meters
            #     dist = sum(distance)
            #     dist_time.append([dist, time])
            # print("Routes: ", dist_time)

            # print(time_till_pickup, distance_till_pickup, distance_till_dropoff)
            # print("Nxt!!")

        return insertion_cost, time_till_pickup

    def dispatch_vehicles(self, commands):
        # print("D: ", commands)
        od_pairs = []
        vehicles = []
        # Comamnd is a dictionary created in dummy_agent
        for command in commands:
            vehicle = VehicleRepository.get(command["vehicle_id"])
            if vehicle is None:
                self.logger.warning("Invalid Vehicle id")
                continue

            if "offduty" in command:
                off_duration = self.sample_off_duration()   # Rand time to rest
                vehicle.take_rest(off_duration)
            elif "cache_key" in command:
                l, a = command["cache_key"]
                route, triptime = self.routing_engine.get_route_cache(l, a)
                vehicle.cruise(route, triptime)
            else:
                vehicles.append(vehicle)
                od_pairs.append((vehicle.get_location(), command["destination"]))

        routes = self.routing_engine.route(od_pairs)

        for vehicle, (route, triptime) in zip(vehicles, routes):
            if triptime == 0:
                continue
            vehicle.cruise(route, triptime)

    def __update_time(self):
        self.__t += self.__dt

    def __populate_new_customers(self):
        new_customers = self.demand_generator.generate(self.__t, self.__dt)
        CustomerRepository.update_customers(new_customers)

    def sample_off_duration(self):
        return np.random.randint(OFF_DURATION / 2, OFF_DURATION * 3 / 2)

    def sample_pickup_duration(self):
        return np.random.exponential(PICKUP_DURATION)

    def get_current_time(self):
        t = self.__t
        return t

    def get_new_requests(self):
        return CustomerRepository.get_new_requests()

    def get_vehicles_state(self):
        return VehicleRepository.get_states()

    def get_vehicles(self):
        return VehicleRepository.get_all()

    def get_customers(self):
        return CustomerRepository.get_all()
        # return [VehicleRepository.get(id) for id in v_ids]

    # def log_score(self):
    #     for vehicle in VehicleRepository.get_all():
    #         score = ','.join(map(str, [self.get_current_time(), vehicle.get_id()] + vehicle.get_score()))
    #         sim_logger.log_score(score)
