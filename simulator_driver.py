#import sys
#sys.path.insert(0, "/path/to/project")


import os
import pandas as pd
import numpy as np
from common.time_utils import get_local_datetime
from common import mesh
from novelties import status_codes
from logger import sim_logger
from simulator.settings import FLAGS
from dummy_agent.agent import Dummy_Agent, DQN_Agent
from dummy_agent import demand_loader, pricing_policy
from central_agent import central_agent, matching_policy
import simulator.simulator as sim
from simulator.models.vehicle.vehicle_repository import VehicleRepository
from novelties import agent_codes
from dqn_agent.dqn_policy import DQNDispatchPolicy, DQNDispatchPolicyLearner
from config.settings import TIMESTEP, MAP_WIDTH, MAP_HEIGHT, ENTERING_TIME_BUFFER,DEFAULT_LOG_DIR
from datetime import datetime
import time

def load():
    setup_base_log_dir(FLAGS.tag)

    if FLAGS.train:
        print("Set training mode")
        # print(tf.__version__)
        dispatch_policy = DQNDispatchPolicyLearner()
        dispatch_policy.build_q_network(load_network=FLAGS.load_network)

        if FLAGS.load_memory:
            # print(FLAGS.load_memory)
            dispatch_policy.load_experience_memory(FLAGS.load_memory)

        if FLAGS.pretrain > 0:
            for i in range(FLAGS.pretrain):
                average_loss, average_q_max = dispatch_policy.train_network(FLAGS.batch_size)
                # print("iterations : {}, average_loss : {:.3f}, average_q_max : {:.3f}".format(
                #     i, average_loss, average_q_max), flush=True)
                dispatch_policy.q_network.write_summary(average_loss, average_q_max)

    else:
        dispatch_policy = DQNDispatchPolicy()
        if FLAGS.load_network:
            dispatch_policy.build_q_network(load_network=FLAGS.load_network)

    return dispatch_policy


def setup_base_log_dir(base_log_dir):
    print("Setup")
    base_log_path = "./logs/{}".format(base_log_dir)
    # print(base_log_path)
    if not os.path.exists(base_log_path):
        os.makedirs(base_log_path)
    for dirname in ["sim"]:
        p = os.path.join(base_log_path, dirname)
        if not os.path.exists(p):
            os.makedirs(p)
    if FLAGS.train:
        for dirname in ["networks", "summary", "memory"]:
            p = os.path.join(base_log_path, dirname)
            # print(p)
            if not os.path.exists(p):
                os.makedirs(p)
    # print(DEFAULT_LOG_DIR)
    if os.path.exists(DEFAULT_LOG_DIR):
        # print(DEFAULT_LOG_DIR)
        # print(base_log_dir)
        os.unlink(DEFAULT_LOG_DIR)
    os.symlink(base_log_dir, DEFAULT_LOG_DIR)

class simulator_driver(object):
    # For DQN
    def __init__(self, start_time, timestep, matching_policy, dispatch_policy, pricing_policy):
    # def __init__(self, start_time, timestep, matching_policy):
        self.simulator = sim.Simulator(start_time, timestep)
        # For DQN
        self.dqn_agent = DQN_Agent(pricing_policy, dispatch_policy)
        self.dummy_agent = Dummy_Agent(pricing_policy, dispatch_policy=None)
        self.central_agent = central_agent.Central_Agent(matching_policy)
        self.last_vehicle_id = 1
        self.vehicle_queue = []

    # Assign vehicles to random initial locations
    def sample_initial_locations(self, t):
        locations = [mesh.convert_xy_to_lonlat(x, y)[::-1] for x in range(MAP_WIDTH) for y in range(MAP_HEIGHT)]
        p = demand_loader.DemandLoader.load_demand_profile(t)
        p = p.flatten() / p.sum()
        vehicle_locations = [locations[i] for i in np.random.choice(len(locations), size=FLAGS.vehicles, p=p)]
        # print("Num: ", len(vehicle_locations))
        return vehicle_locations

    # Populate vehicles over the city map.
    def populate_vehicles(self, vehicle_locations):
        n_vehicles = len(vehicle_locations)
        vehicle_ids = range(self.last_vehicle_id, self.last_vehicle_id + n_vehicles)
        self.last_vehicle_id += n_vehicles

        t = self.simulator.get_current_time()
        entering_time = np.random.uniform(t, t + ENTERING_TIME_BUFFER, n_vehicles).tolist()
        q = sorted(zip(entering_time, vehicle_ids, vehicle_locations))
        self.vehicle_queue = q

    # Initialize vehicles into the city according to their scheduled time to enter the market.
    def enter_market(self):
        t = self.simulator.get_current_time()
        while self.vehicle_queue:
            t_enter, vehicle_id, location = self.vehicle_queue[0]
            if t >= t_enter:
                self.vehicle_queue.pop(0)
                self.simulator.populate_vehicle(vehicle_id, location)
            else:
                break


if __name__ == '__main__':

    start = time.time()
    # For DQN
    dispatch_policy = load()
    # setup_base_log_dir(FLAGS.tag)
    if FLAGS.days > 0:
        start_time = FLAGS.start_time + int(60 * 60 * 24 * FLAGS.start_offset)
        # print(start_time, MAX_DISPATCH_CYCLE, MIN_DISPATCH_CYCLE)
        print("Start Datetime: {}".format(get_local_datetime(start_time)))
        end_time = start_time + int(60 * 60 * 24 * FLAGS.days)
        print("End Datetime  : {}".format(get_local_datetime(end_time)))

        # For DQN
        Sim_experiment = simulator_driver(start_time, TIMESTEP, matching_policy.GreedyMatchingPolicy(), dispatch_policy, pricing_policy.PricingPolicy())
        # Sim_experiment = simulator_driver(start_time, TIMESTEP, matching_policy.GreedyMatchingPolicy())

        # header = "TimeStamp, Unix TIme, Vehicles, Occupied Vehicles, Requests, Matchings, Rejects, Accepts, Avg Wait Time per request, Avg Earnings, " \
        #          "Avg Cost, Avg Profit for DQN, Avg Profit for dummy, Avg Total Dist, Avg Capacity per vehicle, Avg Idle Time"
        # sim_logger.log_summary(header)
        # header = "TimeStamp, Request ID, Status Code, Waiting_Time"
        # sim_logger.log_customer_event(header)
        # header = "V_id', V_lat, V_lon, Speed, Status, Dest_lat, Dest_lon, Type, Travel_Dist, Price_per_travel_m, Price_per_wait_min, Gas_price,"\
        # "assigned_customer_id, Time_to_destination, Idle_Duration, Total_Idle, Current_Capacity, Max_Capacity, Driver_base_per_trip, Mileage"
        # sim_logger.log_vehicle_event(header)

        n_steps = int(3600 * 24 / TIMESTEP)
        buffer_steps = int(3600 / TIMESTEP)
        # print(DB_HOST_PATH)
        # n = 0
        for _ in range(FLAGS.days):
            vehicle_locations = Sim_experiment.sample_initial_locations(Sim_experiment.simulator.get_current_time() + 3600 * 3)
            Sim_experiment.populate_vehicles(vehicle_locations)
            sum_avg_cust = 0
            # sum_avg_profit = 0
            sum_avg_wait = 0
            sum_requests = 0
            sum_accepts = 0
            sum_rejects = 0
            prev_rejected_req = []
            print("############################ SUMMARY ################################")
            for i in range(n_steps):
                Sim_experiment.enter_market()
                Sim_experiment.simulator.step()
                vehicles = Sim_experiment.simulator.get_vehicles_state()
                # print("V: ", len(vehicles))
                requests = Sim_experiment.simulator.get_new_requests()
                col_names = requests.columns.values
                # print("Col: ", requests.columns.values)
                # print("R before: ", len(requests))
                # print(len(prev_rejected_req))

                if FLAGS.enable_pricing:
                    prev_df = pd.DataFrame()
                    for r in prev_rejected_req:
                        r_df = pd.DataFrame({str(c):[float(getattr(r, c))] for c in col_names})
                        # print("B: ", len(requests), len(r_df))
                        requests = requests.append(r_df, ignore_index=True)

                sum_requests += len(requests)
                requests = requests.set_index("id")
                # print("R After: ", len(requests))

                current_time = Sim_experiment.simulator.get_current_time()

                # For DQN
                if FLAGS.enable_pricing:
                    # print("All: ", len(vehicles), m)
                    if len(vehicles) > 0:
                        new_vehicles = vehicles.loc[[vehicle_id for vehicle_id in vehicles.index
                        if VehicleRepository.get(vehicle_id).first_dispatched == 0]]
                        startup_dispatch = Sim_experiment.dqn_agent.startup_dispatch(current_time, new_vehicles)
                        Sim_experiment.simulator.dispatch_vehicles(startup_dispatch)
                        # print("Done", len(new_vehicles))

                if len(vehicles) == 0:
                    continue
                else:
                    # print("V1: ", len(vehicles))
                    m_commands, vehicles, num_matched = Sim_experiment.central_agent.get_match_commands(current_time,
                                                                                                 vehicles, requests)
                    # print("V2: ", len(vehicles))

                    # V_R_matching = defaultdict(list)
                    # for command in m_commands:
                    #     V_R_matching[command["vehicle_id"]].append(CustomerRepository.get(command["customer_id"]).get_request())

                    dqn_v = vehicles[vehicles.agent_type == agent_codes.dqn_agent]
                    dummy_v = vehicles[vehicles.agent_type == agent_codes.dummy_agent]
                    # print("DQN: ", len(dqn_v), " Dummy: ", len(dummy_v))

                    # For DQN and Dummy
                    d1_commands = Sim_experiment.dummy_agent.get_dispatch_commands(current_time, dummy_v)
                    d2_commands = Sim_experiment.dqn_agent.get_dispatch_commands(current_time, dqn_v)
                    # print("1: ", len(d1_commands), " 2: ", len(d2_commands))

                    all_commands = d1_commands + d2_commands
                    # print("A: ", len(all_commands), " 1: ", len(d1_commands), " 2: ", len(d2_commands))

                    prev_rejected_req, accepted_commands, num_accepted = Sim_experiment.simulator.match_vehicles(
                        m_commands, Sim_experiment.dqn_agent, Sim_experiment.dummy_agent)
                    # For DQN
                    Sim_experiment.simulator.dispatch_vehicles(all_commands)

                    if (num_matched == 0):
                        # print("ERR!", len(vehicles), len(requests), num_accepted, len(prev_rejected_req))
                        continue

                    avg_cap = 0
                    capacity = []
                    for index, v in vehicles.iterrows():
                        if v.status == status_codes.V_OCCUPIED:
                            capacity.append(v.current_capacity)
                    if len(capacity):
                        avg_cap = np.sum(capacity) / len(capacity)
                        sum_avg_cust += avg_cap
                        # print(len(capacity), sum_avg_cust)

                    net_v = vehicles[vehicles.status != status_codes.V_OFF_DUTY]
                    # print(len(net_v))
                    occ_v = net_v[net_v.status == status_codes.V_OCCUPIED]

                    if len(occ_v) != len(capacity):
                        print("Watch Occupied", len(occ_v), len(capacity))

                    if FLAGS.enable_pricing:
                        if len(accepted_commands) > 0:
                            average_wt = np.mean([np.mean(list(command['duration'])) for command in
                                                  accepted_commands]).astype(float)
                        else:
                            average_wt = 0
                    else:
                        if num_matched > 0:
                            average_wt = np.mean([command['duration'] for command in m_commands]).astype(int)
                        else:
                            average_wt = 0

                    # sum_avg_wait += average_wt

                    # Start time is a unix timesatmp, here we convert it to normal time
                    readable_time = datetime.utcfromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
                    if FLAGS.enable_pricing:
                        rejected_requests = len(requests) - num_accepted
                        sum_accepts += num_accepted
                    else:
                        rejected_requests = len(requests) - num_matched
                        sum_accepts += num_matched
                    # print("Total Rejected: ", rejected_requests)
                    sum_rejects += rejected_requests

                    avg_total_dist = np.mean(list(v.travel_dist for index, v in net_v.iterrows()))
                    avg_idle_time = np.mean(list(v.total_idle for index, v in net_v.iterrows()))
                    # avg_idle_time = np.mean(list(v.get_idle_duration() for index, v in net_v.iterrows()))
                    avg_earnings = np.mean(list(v.earnings for index, v in net_v.iterrows()))
                    avg_cost = np.mean(list(v.cost for index, v in net_v.iterrows()))

                    if len(dqn_v) > 0:
                        avg_profit_dqn = np.mean(list(v.earnings - v.cost for index, v in dqn_v.iterrows()))
                    else:
                        avg_profit_dqn = 0

                    if len(dummy_v) > 0:
                        avg_profit_dummy = np.mean(list(v.earnings - v.cost for index, v in dummy_v.iterrows()))
                    else:
                        avg_profit_dummy = 0

                    matchings = np.sum([len(command["customer_id"]) for command in m_commands])
                    # print("P: ", avg_earnings-avg_cost, " P-DQN: ", avg_profit_dqn, " P-D: ", avg_profit_dummy)

                    # if average_wt != float(0):
                    #     average_wt /= len(accepted_commands)

                    summary = "{:s},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}, {:.2f}, {:.2f}".format(
                            readable_time, current_time, len(net_v), len(net_v[net_v.status == status_codes.V_OCCUPIED]), len(requests), num_matched,
                            rejected_requests, num_accepted, average_wt, avg_earnings, avg_cost, avg_profit_dqn, avg_profit_dummy, avg_total_dist,
                            avg_cap, avg_idle_time)

                    sim_logger.log_summary(summary)

                    if FLAGS.verbose:
                        print("summary: ({})".format(summary), flush=True)


    if FLAGS.train:
        print("Dumping experience memory as pickle...")
        dispatch_policy.dump_experience_memory()

