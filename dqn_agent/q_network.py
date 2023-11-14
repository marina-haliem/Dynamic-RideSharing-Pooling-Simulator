import os
from typing import List
from simulator import settings
from simulator.settings import FLAGS
from config.settings import BASE_PATH
from logger import sim_logger

# Numerical imports
import numpy as np

# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense

import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental import checkify

from jax import config

config.update("jax_debug_nans", True)
config.update("jax_disable_jit", True)

import pickle
import tqdm


@jax.jit
def mse_loss(y_values, q_values):
    return jnp.mean(jnp.square(y_values - q_values))


# Standrad Implementation of DeepQNetworks "Parent Class"
# STATE SPACE STATE_FEATURES x ACTION_FEATURES => REWARD
class DeepQNetwork(hk.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__(name="DeepQNetwork")
        #
        # self.logger = sim_logger.getLogger("QNNetwork")
        # self.sa_input, self.q_values = self.build_q_network()
        # self.network_params = hk.data_structures.Params(self.build_q_network())

        # if network_path:
        #     print("Net:", network_path)
        #     self.load_network(network_path)

    def __call__(self, sa_input):
        x = hk.Linear(100, name="dense_1", with_bias=True)(sa_input)
        x = jax.nn.relu(x)
        x = hk.Linear(100, name="dense_2", with_bias=True)(x)
        x = jax.nn.relu(x)
        return hk.Linear(1, name="q_value", with_bias=True)(x)


def compute_q_values(sa_input):
    # s_feature = jnp.array(s_feature)
    # What was intended here was to concat everything into the input!
    # sa_input = s_feature + a_features
    q_values = DeepQNetwork()(sa_input)
    return q_values


# Get action associated with max q-value
@jax.jit
def get_action(q_values, amax):
    if FLAGS.alpha > 0:
        exp_q = jnp.exp((q_values - q_values[amax]) / FLAGS.alpha)
        p = exp_q / exp_q.sum()
        return np.random.choice(len(p), p=p)
    else:
        return amax


# Get price associated with max q-value
@jax.jit
def get_price(q_values, amax):
    if FLAGS.alpha > 0:
        exp_q = jnp.exp((q_values - q_values[amax]) / FLAGS.alpha)
        p = exp_q / exp_q.sum()
        return np.random.choice(len(p), p=p)
    else:
        return amax


@jax.jit
def UpdateWeights(weights, gradients, learning_rate):
    return weights - learning_rate * gradients


def save(ckpt_dir: str, state) -> None:
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    with open(os.path.join(ckpt_dir, "arrays.npy"), "wb") as f:
        for x in jax.tree_leaves(state):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, state)
    with open(os.path.join(ckpt_dir, "tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)

    print("Successfully saved: " + save_path)


def restore(ckpt_dir):
    with open(os.path.join(ckpt_dir, "tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)

    leaves, treedef = jax.tree_flatten(tree_struct)
    with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return jax.tree_unflatten(treedef, flat_state)


class DeepQTrainingLoop:
    def __init__(self, wasInstantiated=False):
        # TODO Feed training data to the model
        self.wasInitiated = wasInstantiated
        self.epsilon = 0.01
        self.n_steps = 0
        self.epsilon = settings.INITIAL_EPSILON
        self.epsilon_step = (
            settings.FINAL_EPSILON - settings.INITIAL_EPSILON
        ) / settings.EXPLORATION_STEPS

    # def load_memory_data(self, data):
    #     assert False
    #     self.training_data = data

    def instatiateNets(self, s_features, a_features, load_prev: bool, ckpt_dir="model"):
        self.rng = jax.random.PRNGKey(42)

        # self.conv_net = hk.transform(CNN)
        self.applyDQN = hk.transform(compute_q_values)

        # paramsCNN = self.conv_net.init(self.rng, self.X_train[:5])
        sa_input = jnp.array([s_features + a_feature for a_feature in a_features])
        self.params_agent = self.applyDQN.init(self.rng, sa_input)
        # paramsClassifier = self.applyClassifier.init(self.rng, test_data)

        _ = self.applyDQN.apply(self.params_agent, self.rng, sa_input)

        self.wasInitiated = True
        if load_prev:
            self.params_agent = self.restore_model(ckpt_dir)

        return self.params_agent

    def restore_model(self, ckpt_dir="model", name="dqn_agent"):
        return restore(BASE_PATH / f"{ckpt_dir}/{name}")

    def training_op(self, params, training_tuples):
        sa_data = jnp.array([t.state_action_features for t in training_tuples])
        rewards = jnp.array([t.reward for t in training_tuples])

        # checkify.check(bool(sa_data.shape), "SHOULD HAVE STATE - ACTION FEATURES")
        # checkify.check(bool(rewards.shape), "SHOULD HAVE REWARDS")

        vApply = jax.vmap(lambda x: self.applyDQN.apply(params, self.rng, x), in_axes=1)
        q_values = vApply(sa_data)
        jax.debug.print("PARAMS ARE => {p}", p=params)
        jax.debug.print("SA_BATCH ARE => {p}", p=sa_data)
        jax.debug.print("REWARDS ARE => {p}", p=rewards)
        jax.debug.print("Q_VALUES ARE => {p}", p=q_values)
        return mse_loss(q_values, rewards)
        # vLoss = jax.vmap(mse_loss, in_axes=(0,0))
        # return vLoss(q_values, rewards)

    def run_cyclic_updates(self, params_agent):
        self.n_steps += 1
        # Update target network
        # if self.n_steps % settings.TARGET_UPDATE_INTERVAL == 0:
        #     self.update_target_network()
        #     print("Update target network")

        if self.n_steps % settings.SAVE_INTERVAL == 0:
            # Note: Saving and loading models in Haiku is usually done outside the module.
            # You can use jax.tree_util.tree_flatten and tree_unflatten to save and load parameters.
            save_path = "model"  # Replace with your saving logic
            save(state=params_agent, ckpt_dir=BASE_PATH / "model/dqn_agent")
            print("Successfully saved: " + save_path)

        # Anneal epsilon linearly over time
        if self.n_steps < settings.EXPLORATION_STEPS:
            self.epsilon += self.epsilon_step

    def training_step(
        self,
        training_tuples,
        learning_rate=1 / 1e4,
        ckpt_dir="model",
    ):
        # params = self.conv_net.init(self.rng, self.X_train[:5])
        print("Starting Training...")
        # if not self.wasInitiated:
        #     params_agent = self.instatiateNets(load_prev=True)

        # First argument must be the weights to take the gradients with respect to!
        losses_agent = []
        # sim_logger.log_dqn(f"TRAINING TUPLES => {training_tuples}")

        # jax.debug.breakpoint()
        # jax.debug.print("training tuples => {y}", y=training_tuples)
        evaluateLossAgent = jax.value_and_grad(
            lambda params: self.training_op(params, training_tuples)
        )
        # TODO this is vmap'ed  over the batch axis
        loss_agent, param_grads_agent = evaluateLossAgent(self.params_agent)

        # self.params_agent = UpdateWeights(
        #     self.params_agent, param_grads_agent, learning_rate
        # )
        self.params_agent = jax.tree_map(
            lambda x, y: UpdateWeights(x, y, learning_rate),
            self.params_agent,
            param_grads_agent,
        )  ## Update Params

        losses_agent.append(loss_agent)  ## Record Loss
        # return losses_agent, self.params_agent
        return losses_agent

    def save_model(self, params_classifier, params_embedder, ckpt_dir="model"):
        save(os.path.join(ckpt_dir, "classifier"), params_classifier)

    # Greedy Approach to get action with max q-value
    def get_action(self, q_values, amax):
        # e-greedy exploration
        if self.epsilon > np.random.random():
            return np.random.randint(len(q_values))
        else:
            return get_action(q_values, amax)

    def get_fingerprint(self):
        return self.n_steps, self.epsilon

    def compute_target_q_values(self, s):
        s_feature, a_features = s
        if not self.wasInitiated:
            sim_logger.log_dqn("INSTATIATING NETS...")
            self.instatiateNets(
                load_prev=False, s_features=s_feature, a_features=a_features
            )
        sim_logger.log_dqn(f"NETS INSTANTIATED PARAMS => {self.params_agent}")
        sim_logger.log_dqn(f"STATE AND ACTION FEATURES => {s_feature} - {a_features}")

        sa_input = jnp.array([s_feature + a_feature for a_feature in a_features])

        q_values = self.applyDQN.apply(self.params_agent, self.rng, sa_input)
        return q_values

    def compute_target_value(self, s):
        Q = self.compute_target_q_values(s)

        s_feature, a_features = s
        sa_input = jnp.array([s_feature + a_feature for a_feature in a_features])

        q_values = self.applyDQN.apply(self.params_agent, self.rng, sa_input)
        amax = jnp.argmax(q_values)
        V = Q[amax]
        if FLAGS.alpha > 0:
            V += FLAGS.alpha * jnp.log(np.exp((Q - Q.max()) / FLAGS.alpha).sum())
        return V

    def setup_summary(self):
        raise NotImplementedError("WE DONT NEED THIS")

    def write_summary(self, avg_loss, avg_q_max):
        # Write your summary logic here
        raise NotImplementedError("Replace this with the logging module")


# # Learner Q-network used in trining mode
# class FittingDeepQNetwork(DeepQNetwork):
#     def __init__(self, network_path=None):
#         super(FittingDeepQNetwork, self).__init__(network_path)

#         self.sa_input, self.q_values = self.build_q_network()
#         self.model_params = hk.data_structures.Params(self.build_q_network())

#         # Create target network
#         self.target_sub_input, self.target_q_values = self.build_q_network()
#         # target_model_params = hk.data_structures.Params(self.build_q_network())
#         # Define target network update operation
#         # self.update_target_network = [jax.tree_multimap(lambda x, y: y, target_model_params, self.model_params)]

#         # Define loss and gradient update operation
#         self.y, self.loss, self.grad_update = self.build_training_op(self.model_params)

#         # Initialize target network
#         self.update_target_network()

#         self.n_steps = 0
#         self.epsilon = settings.INITIAL_EPSILON
#         self.epsilon_step = (
#             settings.FINAL_EPSILON - settings.INITIAL_EPSILON
#         ) / settings.EXPLORATION_STEPS

#         for var_name, var_value in hk.data_structures.dataclass_items(
#             self.model_params
#         ):
#             # Print or log Haiku parameter names
#             print(f"Parameter Name: {var_name}, Shape: {var_value.shape}")
