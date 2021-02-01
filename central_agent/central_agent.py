from collections import defaultdict

from novelties import status_codes
from simulator.settings import FLAGS
from simulator.models.vehicle.vehicle_repository import VehicleRepository
from simulator.models.customer.customer_repository import CustomerRepository
from common.geoutils import great_circle_distance
from novelties.pricing.price_calculator import calculate_price
from simulator.services.routing_service import RoutingEngine
from common import geoutils

class Central_Agent(object):

    def __init__(self, matching_policy):
        self.matching_policy = matching_policy
        self.routing_engine = RoutingEngine.create_engine()

    def get_match_commands(self, current_time, vehicles, requests):
        matching_commands = []
        num_matched = 0
        if len(requests) > 0:
            if FLAGS.enable_pooling:
                matching_commands = self.matching_policy.match_RS(current_time, vehicles, requests)
            else:
                matching_commands = self.matching_policy.match(current_time, vehicles, requests)

            num_matched = len(matching_commands)
            updated_commands = self.init_price(matching_commands)

            final_commands = []
            if FLAGS.enable_pooling:

                V = defaultdict(list)
                # V_pair = defaultdict(list)
                V_duration = defaultdict(list)
                V_price = defaultdict(list)
                V_dist = defaultdict(list)
                # V_dest = defaultdict(list)

                for command in updated_commands:
                    V[command["vehicle_id"]].append(command["customer_id"])
                    # pair_o, pair_d = pair
                    # V_pair[command["vehicle_id"]].append(pair_d)
                    # customer = CustomerRepository.get(command["customer_id"])
                    # V_dest[command["vehicle_id"]].append(customer.get_destination())
                    V_duration[command["vehicle_id"]].append(command["duration"])
                    V_price[command["vehicle_id"]].append(command["init_price"])
                    V_dist[command["vehicle_id"]].append(command["distance"])

                for k in V.keys():
                    # print(k)
                    new_command = dict()
                    new_command["vehicle_id"] = k
                    new_command["customer_id"] = V[k]
                    # new_command["pickups"] = V_pair[k]
                    new_command["duration"] = V_duration[k]
                    new_command["distance"] = V_dist[k]
                    new_command["init_price"] = V_price[k]
                    # new_command["destinations"] = V_dest[k]
                    final_commands.append(new_command)

            else:
                final_commands = updated_commands
                # print("List: ", len(final_commands))
                # V = defaultdict(list)
                # for command in matching_commands:
                #     # print(command)
                #     V[command["vehicle_id"]].append(command["customer_id"])
                #
                # for k in V.keys():
                #     vehicle = VehicleRepository.get(k)
                #     print("ID: ", k, "-> Loc: ", vehicle.get_location())
                #     print(k, "Customers :", V[k])
                #     for i in V[k]:
                #         customer = CustomerRepository.get(i)
                #         print("C_Loc: ", customer.get_origin())
                # # for k in V_pair.keys():
                #     print(k, "Pairs: ", V_pair[k])

            vehicles = self.update_vehicles(vehicles, matching_commands)
            # print(final_commands)
        else:
            return [], vehicles
        return final_commands, vehicles, num_matched


    def update_vehicles(self, vehicles, commands):
        vehicle_ids = [command["vehicle_id"] for command in commands]
        # v = vehicles[vehicles.status != status_codes.V_OCCUPIED]
        # for vid in vehicle_ids:
        #     v = VehicleRepository.get(vid)
            # if (v.state.status == status_codes.V_OCCUPIED) & v.state.accept_new_request:
            #     print("Update", v.current_plan)
            #     if (len(v.current_plan) == 0) & (v.get_destination is not None):
            #         v.current_plan = [v.get_destination()]
            #         v.current_plan_routes = [(v.get_route(), v.state.time_to_destination)]
            #
            #     elif (len(v.current_plan) != 0) & (
            #             v.get_destination() != v.current_plan[0]):
            #         plan = [v.get_destination()]
            #         plan.extend(v.current_plan)
            #         v.current_plan = plan
            #
            #         routes = [(v.get_route(), v.state.time_to_destination)]
            #         routes.extend(v.current_plan_routes)
            #         v.current_plan_routes = routes
            #
            #     print(vid, ":", len(v.current_plan), len(v.current_plan_routes), len(v.ordered_pickups_dropoffs_ids),
            #           len(v.pickup_flags))
        vehicles.loc[vehicle_ids, "status"] = status_codes.V_ASSIGNED
        # print(vehicles.loc[vehicle_ids,"status"])
        return vehicles


    def init_price(self, match_commands):
        m_commands = []
        for c in match_commands:
            vehicle = VehicleRepository.get(c["vehicle_id"])
            # vehicle.state.status = status_codes.V_ASSIGNED
            if vehicle is None:
                print("Invalid Vehicle id")
                continue
            customer = CustomerRepository.get(c["customer_id"])
            if customer is None:
                print("Invalid Customer id")
                continue

            triptime = c["duration"]

            # if FLAGS.enable_pricing:
            dist_for_pickup = c["distance"]

            od_route = self.routing_engine.route_time([(customer.get_origin(), customer.get_destination())])
            # dist_dropoff = great_circle_distance(customer.get_origin()[0], customer.get_origin()[1],
            #                               customer.get_destination()[0], customer.get_destination()[1])
            route, time = od_route[0]
            lats, lons = zip(*route)
            distance = geoutils.great_circle_distance(lats[:-1], lons[:-1], lats[1:], lons[1:])  # Distance in meters
            dist_till_dropoff = sum(distance)

            # print(dist_dropoff, dist_till_dropoff)

            total_trip_dist = dist_for_pickup + dist_till_dropoff
            [travel_price, wait_price] = vehicle.get_price_rates()
            # v_cap = vehicle.state.current_capacity
            initial_price = calculate_price(total_trip_dist, triptime, vehicle.state.mileage, travel_price,
                                wait_price, vehicle.state.gas_price, vehicle.state.driver_base_per_trip)
            c["init_price"] = initial_price
            m_commands.append(c)
            # print(c)

        return m_commands