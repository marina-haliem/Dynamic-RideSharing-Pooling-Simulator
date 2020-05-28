from simulator.settings import FLAGS, INITIAL_MEMORY_SIZE
from simulator.models.vehicle.vehicle_repository import VehicleRepository


class Dummy_Agent(object):

    def __init__(self, pricing_policy, dispatch_policy):
        self.pricing_policy = pricing_policy
        self.dispatch_policy = dispatch_policy

    def get_dispatch_commands(self, current_time, vehicles):

        # dispatch_commands = self.dispatch_policy.dispatch(current_time, vehicles)
        # return dispatch_commands
        return []

    def get_price_decision(self, vehicle, price, request):
        response = self.pricing_policy.propose_price(vehicle, price, request)

        return response

class DQN_Agent(Dummy_Agent):

    def get_dispatch_commands(self, current_time, vehicles):
        dispatch_commands = self.dispatch_policy.dispatch(current_time, vehicles)
        return dispatch_commands

    def startup_dispatch(self, current_time, vehicles):
        self.dispatch_policy.update_state(current_time, vehicles)
        self.dispatch_policy.give_rewards(vehicles)
        dispatch_commands = self.dispatch_policy.get_dispatch_decisions(vehicles)
        self.dispatch_policy.record_dispatch(vehicles.index, current_time)
        for vid in vehicles.index:
            vehicle = VehicleRepository.get(vid)
            vehicle.first_dispatched = 1
        ######### Only with the DQN agent not the dummy #############
        self.dispatch_policy.backup_supply_demand()
        # If size exceeded, run training
        if len(self.dispatch_policy.supply_demand_history) > INITIAL_MEMORY_SIZE:
            average_loss, average_q_max = self.dispatch_policy.train_network(FLAGS.batch_size)
            # print("iterations : {}, average_loss : {:.3f}, average_q_max : {:.3f}".format(
            #     self.q_network.n_steps, average_loss, average_q_max), flush=True)
            self.dispatch_policy.q_network.write_summary(average_loss, average_q_max)
        return dispatch_commands


