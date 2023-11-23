import os
import logging.config
from logging import getLogger
import yaml

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging.yaml")


class SimulationLogger(object):
    def setup_logging(self, env, path=config_path, level=logging.INFO):
        print("SETTING UP LOGGER")
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
        self.vehicle_logger = getLogger("vehicle")
        self.customer_logger = getLogger("customer")
        self.summary_logger = getLogger("summary")
        # self.avg_summary = getLogger('avg_summary')
        self.score_logger = getLogger("score")
        self.dqn_logger = getLogger("dqn_agent")
        self.env = env

    def get_current_time(self):
        if self.env:
            return self.env.get_current_time()
        return 0

    def log_vehicle_event(self, msg):
        t = self.get_current_time()
        self.log_dqn("{},{}".format(str(t), msg))
        self.vehicle_logger.info("{},{}".format(str(t), msg))

    def log_customer_event(self, msg):
        t = self.get_current_time()
        self.customer_logger.info("{},{}".format(str(t), msg))

    def log_summary(self, summary):
        self.summary_logger.info(summary)

    def log_dqn(self, val, level="info"):
        getattr(self.dqn_logger, level)(val)
        # self.dqn_logger.info(val)

    def log_score(self, score):
        self.score_logger.info(score)


sim_logger = SimulationLogger()
