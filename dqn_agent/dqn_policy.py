import pickle
import os
import numpy as np
from collections import OrderedDict, defaultdict
from simulator.settings import FLAGS
from config.settings import GLOBAL_STATE_UPDATE_CYCLE, MIN_DISPATCH_CYCLE
from dqn_agent.feature_constructor import FeatureConstructor
from dqn_agent.q_network import DeepQNetwork, FittingDeepQNetwork
from dummy_agent.dispatch_policy import DispatchPolicy
from simulator import settings
from common.time_utils import get_local_datetime
from common import mesh
from novelties import status_codes
from simulator.models.vehicle.vehicle_repository import VehicleRepository

class DQNDispatchPolicy(DispatchPolicy):
    def __init__(self):
        super().__init__()
        self.feature_constructor = FeatureConstructor()
        self.q_network = None
        self.q_cache = {}


    def build_q_network(self, load_network=None):
        self.q_network = DeepQNetwork(load_network)

    # Overriding the parent function in dummy_agent.dispatch_policy
    def update_state(self, current_time, vehicles):
        t = self.feature_constructor.get_current_time()
        if t == 0 or current_time % GLOBAL_STATE_UPDATE_CYCLE == 0:
            self.q_cache = {}
            # Updating Supply and Demand
            self.feature_constructor.update_supply(vehicles)
            self.feature_constructor.update_demand(current_time)

        self.feature_constructor.update_time(current_time)

    # Overriding the parent function in dummy_agent.dispatch_policy
    def get_dispatch_decisions(self, tbd_vehicles):
        dispatch_commands = []
        for vehicle_id, vehicle_state in tbd_vehicles.iterrows():
            # Get best action for this vehicle and whether it will be offduty or not
            a, offduty = self.predict_best_action(vehicle_id, vehicle_state)
            if offduty:
                command = self.create_dispatch_dict(vehicle_id, offduty=True)
            else:
                # Get target destination and key to cache
                target, cache_key = self.convert_action_to_destination(vehicle_state, a)
                # create dispatch dictionary with the given attribute
                if target is None:
                    continue
                if cache_key is None:
                    command = self.create_dispatch_dict(vehicle_id, target)
                else:
                    command = self.create_dispatch_dict(vehicle_id, cache_key=cache_key)
            dispatch_commands.append(command)
        return dispatch_commands

    # Return best action for this vehicle given its state, and returns whether it will be Offduty or not
    def predict_best_action(self, vehicle_id, vehicle_state):
        if vehicle_state.idle_duration >= MIN_DISPATCH_CYCLE and FLAGS.offduty_probability > np.random.random():
            a, offduty = (0, 0), 1

        elif self.q_network is None:
            a, offduty = (0, 0), 0

        else:
            x, y = mesh.convert_lonlat_to_xy(vehicle_state.lon, vehicle_state.lat)
            if (x, y) in self.q_cache:
                actions, Q, amax = self.q_cache[(x, y)]
            else:
                # Get state features and action features
                s, actions = self.feature_constructor.construct_current_features(x, y)
                # print("A: ", actions, len(actions))
                # print("S: ", s)
                # Calculate Q values based on state features
                Q = self.q_network.compute_q_values(s)
                # print("Q values: ", Q)
                # only considers actions whose values are greater than wait action value
                wait_action_value = Q[0]
                actions = [a for a, q in zip(actions, Q) if q >= wait_action_value]
                Q = Q[Q >= wait_action_value]
                # print("Q ba: ", Q)
                amax = np.argmax(Q)     # Get the index of the max value
                # print("Q max, val, max: ", amax, Q[amax], max(Q))
                self.q_cache[(x, y)] = actions, Q, amax     # Save in cache
            # if actions[amax] == (0, 0):
            #     aidx = amax
            # else:
            aidx = self.q_network.get_action(Q, amax)   # Get action with max Q value
            a = actions[aidx]
            offduty = 1 if Q[aidx] < FLAGS.offduty_threshold else 0
            # print("Chosen A: ", a)
            vehicle = VehicleRepository.get(vehicle_id)
            # tmp_q_action = {a:q for a, q in zip(actions, Q)}
            q_action = {(x+ax, y+ay):q for (ax,ay), q in zip(actions, Q)}
            vehicle.q_action_dict = q_action
            vehicle.epsilon = int(len(vehicle.q_action_dict)*0.05)
            # print("Loc: ", x, " , ", y)
            # print("Act: ", tmp_q_action)
            # print("Added:", vehicle.q_action_dict)
        return a, offduty

    # Get the destination from dispatched vehicles
    def convert_action_to_destination(self, vehicle_state, a):
        cache_key = None
        target = None
        ax, ay = a  # Action from action space matrix
        x, y = mesh.convert_lonlat_to_xy(vehicle_state.lon, vehicle_state.lat)
        lon, lat = mesh.convert_xy_to_lonlat(x + ax, y + ay)
        if lon == vehicle_state.lon and lat == vehicle_state.lat:
            pass
        elif FLAGS.use_osrm and mesh.convert_xy_to_lonlat(x, y) == (lon, lat):
            cache_key = ((x, y), (ax, ay))  # Create cache key with location associated with action
        else:
            target = (lat, lon)

        return target, cache_key


# Learner that uses experience memory and learned previous models
class DQNDispatchPolicyLearner(DQNDispatchPolicy):
    def __init__(self):
        super().__init__()
        self.supply_demand_history = OrderedDict()
        self.experience_memory = []
        self.last_state_actions = {}
        self.rewards = defaultdict(int)
        self.last_earnings = defaultdict(int)
        self.last_cost = defaultdict(int)


    def reset(self):
        self.last_state_actions = {}
        self.rewards = defaultdict(int)
        self.last_earnings = defaultdict(int)
        self.last_cost = defaultdict(int)

    # Store memory
    def dump_experience_memory(self):
        sd_path = os.path.join(FLAGS.save_memory_dir, "sd_history.pkl")
        sars_path = os.path.join(FLAGS.save_memory_dir, "sars_history.pkl")
        pickle.dump(self.supply_demand_history, open(sd_path, "wb"))
        pickle.dump(self.experience_memory, open(sars_path, "wb"))

    # Load stored memory
    def load_experience_memory(self, path):
        sd_path = os.path.join(path, "sd_history.pkl")
        sars_path = os.path.join(path, "sars_history.pkl")
        self.supply_demand_history = pickle.load(open(sd_path, "rb"))
        self.experience_memory = pickle.load(open(sars_path, "rb"))
        # print(len(self.experience_memory))
        state_action, _, _ = self.experience_memory[0]
        t_start, _, _ = state_action
        state_action, _, _ = self.experience_memory[-1]
        t_end, _, _ = state_action
        print("period: {} ~ {}".format(get_local_datetime(t_start), get_local_datetime(t_end)))


    def build_q_network(self, load_network=None):
        self.q_network = FittingDeepQNetwork(load_network)


    def predict_best_action(self, vehicle_id, vehicle_state):
        a, offduty = super().predict_best_action(vehicle_id, vehicle_state)

        if not offduty and settings.WAIT_ACTION_PROBABILITY * self.q_network.epsilon > np.random.random():
            a = (0, 0)      # Init action to be taken if not specified

        self.memorize_experience(vehicle_id, vehicle_state, a)
        return a, offduty

    # Calculate Vehicle Reward
    def give_rewards(self, vehicles):
        for vehicle_id, row in vehicles.iterrows():
            vehicle = VehicleRepository.get(vehicle_id)
            earnings = row.earnings - self.last_earnings.get(vehicle_id, 0)
            cost = vehicle.compute_fuel_consumption()
            if earnings > 0:
                cost -= self.last_cost.get(vehicle_id, 0)
                profit = earnings - cost
            else:
                profit = earnings
            # print("Earnings: ", earnings, "Cost: ", cost, "Profit: ", profit)
            self.rewards[vehicle_id] += (12 * profit) - 5 * row.pickup_time + 10 * settings.STATE_REWARD_TABLE[
                row.status]
            # self.rewards[vehicle_id] += earnings + settings.STATE_REWARD_TABLE[row.status]
            self.last_earnings[vehicle_id] = row.earnings
            if earnings > 0:
                self.last_cost[vehicle_id] = vehicle.compute_fuel_consumption()
            if row.status == status_codes.V_OFF_DUTY:
                self.last_state_actions[vehicle_id] = None

    def dispatch(self, current_time, vehicles):
        self.give_rewards(vehicles)
        dispatch_commands = super().dispatch(current_time, vehicles)
        self.backup_supply_demand()
        # If size exceeded, run training
        if len(self.supply_demand_history) > settings.INITIAL_MEMORY_SIZE:
            average_loss, average_q_max = self.train_network(FLAGS.batch_size)
            # print("iterations : {}, average_loss : {:.3f}, average_q_max : {:.3f}".format(
            #     self.q_network.n_steps, average_loss, average_q_max), flush=True)
            self.q_network.write_summary(average_loss, average_q_max)
        return dispatch_commands


    def backup_supply_demand(self):
        current_time = self.feature_constructor.get_current_time()

        if current_time % GLOBAL_STATE_UPDATE_CYCLE == 0:
            f = self.q_network.get_fingerprint()
            self.feature_constructor.update_fingerprint(f)
            # Load supply demand maps with fingerprint
            self.supply_demand_history[current_time] = self.feature_constructor.get_supply_demand_maps(), f

            if len(self.supply_demand_history) > settings.NUM_SUPPLY_DEMAND_HISTORY:
                self.supply_demand_history.popitem(last=False)

    # Replay when needed (Returns map and fingerprint)
    def replay_supply_demand(self, t):
        t_ = t - (t % GLOBAL_STATE_UPDATE_CYCLE)
        if t_ in self.supply_demand_history:
            return self.supply_demand_history[t_]
        else:
            return None, None

    def memorize_experience(self, vehicle_id, vehicle_state, a):
        t = self.feature_constructor.get_current_time()     # Time
        l = mesh.convert_lonlat_to_xy(vehicle_state.lon, vehicle_state.lat)     # Location
        last_state_action = self.last_state_actions.get(vehicle_id, None)       # Load last action

        if last_state_action is not None:
            current_state = (t, l)
            reward = self.rewards[vehicle_id]

            if len(self.experience_memory) > settings.MAX_MEMORY_SIZE:
                self.experience_memory.pop(0)
            self.experience_memory.append((last_state_action, current_state, reward))

        self.rewards[vehicle_id] = 0    # Reset reward
        self.last_state_actions[vehicle_id] = (t, l, a)     # Update last action

    def train_network(self, batch_size, n_iterations=1):
        loss_sum = 0
        q_max_sum = 0
        for _ in range(n_iterations):
            sa_batch = []
            y_batch = []
            for _ in range(batch_size):
                sa, y = self.replay_memory()    # Get state and action features, with the reward
                sa_batch.append(sa)
                y_batch.append(y)

            loss_sum += self.q_network.fit(sa_batch, y_batch)   # Train model
            q_max_sum += np.mean(y_batch)
        self.q_network.run_cyclic_updates()     # Update target network
        return loss_sum / n_iterations, q_max_sum / n_iterations

    # Replay when needed, returns State features and action features along with reward
    def replay_memory(self, max_retry=100):
        for _ in range(max_retry):
            num = np.random.randint(0, len(self.experience_memory) - 1)
            state_action, next_state, reward = self.experience_memory[num]
            tm, Loc, act = state_action
            next_t, next_l = next_state
            if self.feature_constructor.reachable_map[next_l] == 0:
                self.experience_memory.pop(num)
                continue
            sd, f = self.replay_supply_demand(tm)   # Get supply demand map, and fingerprint
            if sd is None:
                self.experience_memory.pop(num)
                continue
            next_sd, next_f = self.replay_supply_demand(next_t) # Get supply demand map, and fingerprint
            if next_sd is None:
                self.experience_memory.pop(num)
                continue
            a_feature = self.feature_constructor.construct_action_feature(tm, Loc, sd, act)
            if a_feature is None:
                self.experience_memory.pop(num)
                continue

            s_feature = self.feature_constructor.construct_state_feature(tm, f, Loc, sd)
            sa = s_feature + a_feature      # State features and action features
            next_sa, _ = self.feature_constructor.construct_features(next_t, next_f, next_l, next_sd)
            target_value = self.q_network.compute_target_value(next_sa)
            discount_factor = settings.GAMMA ** int((next_t - tm) / 60)
            y = reward + discount_factor * target_value
            return sa, y

        raise Exception
