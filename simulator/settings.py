# from config.settings import MAP_WIDTH, MAP_HEIGHT
from novelties import status_codes
from config.settings import DEFAULT_LOG_DIR, GLOBAL_STATE_UPDATE_CYCLE
import os

# import tensorflow as tf


class flags:
    def __init__(self) -> None:
        pass

    def DEFINE_float(self, name, value, description):
        value = float(value)
        setattr(self, name, value)
        setattr(self, f"{name}_description", description)

    def DEFINE_string(self, name, value, description):
        value = str(value)
        setattr(self, name, value)
        setattr(self, f"{name}_description", description)

    def DEFINE_boolean(self, name, value, description):
        value = bool(value)
        setattr(self, name, value)
        setattr(self, f"{name}_description", description)

    def DEFINE_integer(self, name, value, description):
        value = int(value)
        setattr(self, name, value)
        setattr(self, f"{name}_description", description)


# flags = tf.compat.v1.flags
FLAGS = flags()


FLAGS.DEFINE_float("offduty_threshold", -float("inf"), "q value off duty threshold")
FLAGS.DEFINE_float(
    "offduty_probability", 0.1, "probability to automatically become off duty"
)
FLAGS.DEFINE_float("alpha", 0.0, "entropy coefficient")

FLAGS.DEFINE_string(
    "save_memory_dir", os.path.join(DEFAULT_LOG_DIR, "memory"), "replay memory storage"
)
FLAGS.DEFINE_string(
    "save_network_dir",
    os.path.join(DEFAULT_LOG_DIR, "networks"),
    "network model directory",
)
FLAGS.DEFINE_string(
    "save_summary_dir",
    os.path.join(DEFAULT_LOG_DIR, "summary"),
    "training summary directory",
)
FLAGS.DEFINE_string("load_network", "", "load saved dqn_agent network.")
FLAGS.DEFINE_string("load_memory", "", "load saved replay memory.")

FLAGS.DEFINE_boolean("train", True, "run training dqn_agent network.")
FLAGS.DEFINE_boolean("verbose", True, "print log verbosely.")

FLAGS.DEFINE_boolean("enable_pooling", True, "Enable RideSharing/CarPooling")
FLAGS.DEFINE_boolean("enable_pricing", True, "Enable Pricing Novelty")

FLAGS.DEFINE_integer("vehicles", 10, "number of vehicles")
FLAGS.DEFINE_integer("dummy_vehicles", 0, "number of vehicles using dummy agent")
FLAGS.DEFINE_integer("dqn_vehicles", 10, "number of vehicles using dqn agent")

FLAGS.DEFINE_integer(
    "pretrain", 0, "run N pretraining steps using pickled experience memory."
)
FLAGS.DEFINE_integer(
    "start_time", 1464753600 + 3600 * 5, "simulation start datetime (unixtime)"
)
FLAGS.DEFINE_integer("start_offset", 0, "simulation start datetime offset (days)")

FLAGS.DEFINE_integer("days", 1, "simulation days")

FLAGS.DEFINE_integer("n_diffusions", 3, "number of diffusion convolution")
FLAGS.DEFINE_integer("batch_size", 128, "number of samples in a batch for SGD")
FLAGS.DEFINE_string("tag", "DARM", "tag used to identify logs")
FLAGS.DEFINE_boolean("log_vehicle", True, "whether to log vehicle states")
FLAGS.DEFINE_boolean("use_osrm", False, "whether to use OSRM")
FLAGS.DEFINE_boolean(
    "average", False, "whether to use diffusion filter or average filter"
)
FLAGS.DEFINE_boolean("trip_diffusion", False, "whether to use trip diffusion")

GAMMA = 0.98  # Discount Factor
MAX_MOVE = 7
NUM_SUPPLY_DEMAND_MAPS = 5
# Attributes defined at runtime!
NUM_FEATURES = (
    7
    + NUM_SUPPLY_DEMAND_MAPS * (1 + (FLAGS.n_diffusions + 1) * 2)
    + FLAGS.n_diffusions * 2
    + FLAGS.trip_diffusion * 4
)

# training hyper parameters
WORKING_COST = 0.2
DRIVING_COST = 0.2
STATE_REWARD_TABLE = {
    status_codes.V_IDLE: -WORKING_COST,
    status_codes.V_CRUISING: -(WORKING_COST + DRIVING_COST),
    status_codes.V_ASSIGNED: -(WORKING_COST + DRIVING_COST),
    status_codes.V_OCCUPIED: -(WORKING_COST + DRIVING_COST),
    status_codes.V_OFF_DUTY: 0.0,
}

WAIT_ACTION_PROBABILITY = 0.70  # wait action probability in epsilon-greedy
EXPLORATION_STEPS = 3000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.01  # Final value of epsilon in epsilon-greedy
INITIAL_MEMORY_SIZE = (
    100  # Number of steps to populate the replay memory before training starts
)
NUM_SUPPLY_DEMAND_HISTORY = 7 * 24 * 3600 / GLOBAL_STATE_UPDATE_CYCLE + 1  # = 1 week
MAX_MEMORY_SIZE = 10000000  # Number of replay memory the dummy_agent uses for training
SAVE_INTERVAL = 100  # The frequency with which the network is saved
TARGET_UPDATE_INTERVAL = 50  # The frequency with which the target network is updated
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
