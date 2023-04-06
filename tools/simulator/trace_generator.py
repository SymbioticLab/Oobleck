import time
import random
import argparse
import logging

class TraceGenerator:
    def __init__(self, total_time, add_prob=None, remove_prob=None, interval=60000, 
                 desired_capacity=16, seed=None, n_removal=0, n_removals=None,
                 ):
        
        self.total_time = total_time
        self.add_prob = add_prob
        self.remove_prob = remove_prob
        self.interval = interval
        self.node_count = 0
        self.desired_capacity = desired_capacity
        self.trace = []
        self.n_removals = n_removals

        # k: node_id, v: time_step(added)
        self.active_nodes = {}
        
        self.one_hour = 60000
        self.current_hour = 0

        self.seed = seed
        if self.seed is not None:
            self.r = random.Random(self.seed)
            # print(f'Using seed: {self.seed}')
        else:
            self.r = random.Random()

        # hourly prob
        if self.add_prob is None:
            # logging.debug("Using default add prob")
            # time.sleep(10)
            self.spot_instance_addition_probability = {
                0: 0.05,
                1: 0.05,
                2: 0.05,
                3: 0.50,
                4: 0.50,
                5: 0.50,
                6: 0.05,
                7: 0.05,
                8: 0.05,
                9: 0.05,
                10: 0.05,
                11: 0.05,
                12: 0.05,
                13: 0.05,
                14: 0.05,
                15: 0.00,
                16: 0.00,
                17: 0.00,
                18: 0.00,
                19: 0.00,
                20: 0.00,
                21: 0.00,
                22: 0.00,
                23: 0.05,
            }
        else:
            self.spot_instance_addition_probability = self.generate_probabilities()

        if self.remove_prob is None:
            # print("Using default remove prob")
            self.spot_instance_removal_probability = {
                0: 0.05,
                1: 0.05,
                2: 0.05,
                3: 0.01,
                4: 0.01,
                5: 0.01,
                6: 0.05,
                7: 0.05,
                8: 0.05,
                9: 0.05,
                10: 0.05,
                11: 0.05,
                12: 0.05,
                13: 0.05,
                14: 0.05,
                15: 0.25,
                16: 0.25,
                17: 0.25,
                18: 0.25,
                19: 0.25,
                20: 0.25,
                21: 0.25,
                22: 0.25,
                23: 0.05,
            }
        else:
            self.spot_instance_removal_probability = self.generate_probabilities()

    def generate_probabilities(self):
        probability = {}
        for hour in range(24):
            probability[hour] = self.r.random()
        return probability

    def add_one_node(self, time_step):
        self.node_count += 1
        self.active_nodes[self.node_count] = time_step
        node_id = f"node{self.node_count}"
        event_str = "{},{},{}".format(time_step, "add", node_id)
        self.trace.append(event_str)

    def add_nodes(self, time_step):
        # Check add prob for capacity times
        # add_nodes = []
        hour_add_prob = self.spot_instance_addition_probability[(time_step // self.one_hour) % 24]
        # print(f"hour_add_prob: {hour_add_prob}")
        for i in range(self.desired_capacity):
            if random.random() < hour_add_prob:
                self.add_one_node(time_step)

    def remove_node(self, time_step):
        remove_nodes = self.find_remove_nodes(time_step)
        for nodeid in remove_nodes:
            node_id = f"node{nodeid}"
            event_str = "{},{},{}".format(time_step, "remove", node_id)
            self.trace.append(event_str)
            # revmoe from active_nodes
            self.active_nodes.pop(nodeid)

    def find_remove_nodes(self, time_step):
        # find a list of remove nodes from active_nodes
        remove_nodes = []
        hour_remove_prob = self.spot_instance_removal_probability[(time_step // self.one_hour) % 24]
        # print(f"hour_remove_prob: {hour_remove_prob}")
        if self.n_removals is not None and (len(self.active_nodes) > self.n_removals):
            # sample n_removals from active_nodes
            remove_nodes = random.sample(self.active_nodes.keys(), self.n_removals)
        else:
            for nodeid, time_step in self.active_nodes.items():
                # check remvoe prob for each node
                if random.random() < hour_remove_prob:
                    remove_nodes.append(nodeid)
        return remove_nodes

    def generate_trace(self):
        time_step = 0
        # add initial nodes
        for i in range(self.desired_capacity):
            self.add_one_node(time_step)

        # loop through time steps (0 to total_time)
        # Adding interval from 0, til total_time hits
        while time_step < self.total_time:
            time_step += self.interval
            # add node
            self.add_nodes(time_step)

            # remove node
            self.remove_node(time_step)

        return self.trace



if __name__ == "__main__":
    # total_time = int(input("Enter the total time steps: "))

    parser = argparse.ArgumentParser(description='Generate trace for spot instances')
    parser.add_argument('--total_time', type=int, default=38760000, help='total time steps')
    parser.add_argument('--desired_capacity', type=int, default=12, help='desired capacity')
    parser.add_argument('--interval', type=int, default=60000, help='interval')
    parser.add_argument('--seed', type=int, default=None, help='seed')
    parser.add_argument('--n_removals', type=int, default=None, help='numbers of removals per hour')
    args = parser.parse_args()

    trace_gen = TraceGenerator(total_time=args.total_time, desired_capacity=args.desired_capacity, 
                               interval=args.interval, seed=args.seed, n_removals=args.n_removals)
    # trace_gen = TraceGenerator(total_time)
    trace = trace_gen.generate_trace()

    for line in trace:
        print(line)
