import time
import random
import argparse
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt

def hour2ms(hour):
    return int( hour * 60 * 60 * 1000 )

class TraceGenerator:
    def __init__(self, total_time, add_prob=None, remove_prob=None, interval=60000, 
                 desired_capacity=15, seed=None, n_removal=0, n_removals=None,
                 max_n_node=30, min_n_node=15, no_adds=False):
        
        self.total_time = total_time
        self.add_prob = add_prob
        self.remove_prob = remove_prob
        self.interval = interval
        self.node_count = 0
        self.desired_capacity = desired_capacity
        self.trace = []
        self.n_removals = n_removals
        self.max_n_node = max_n_node
        self.min_n_node = min_n_node
        self.no_adds = no_adds

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
                3: 0.05,
                4: 0.05,
                5: 0.05,
                6: 0.05,
                7: 0.05,
                8: 0.05,
                9: 0.05,
                10: 0.05,
                11: 0.05,
                12: 0.05,
                13: 0.05,
                14: 0.05,
                15: 0.05,
                16: 0.05,
                17: 0.05,
                18: 0.05,
                19: 0.05,
                20: 0.05,
                21: 0.05,
                22: 0.05,
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
                3: 0.05,
                4: 0.05,
                5: 0.05,
                6: 0.05,
                7: 0.05,
                8: 0.05,
                9: 0.05,
                10: 0.05,
                11: 0.05,
                12: 0.05,
                13: 0.05,
                14: 0.05,
                15: 0.05,
                16: 0.05,
                17: 0.05,
                18: 0.05,
                19: 0.05,
                20: 0.05,
                21: 0.05,
                22: 0.05,
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
        if len(self.active_nodes)==self.max_n_node:
            return

        self.node_count += 1
        self.active_nodes[self.node_count] = time_step
        node_id = f"node{self.node_count}"
        event_str = "{},{},{}".format(int(time_step), "add", node_id)
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
            if len(self.active_nodes)==self.min_n_node:
                return
            node_id = f"node{nodeid}"
            event_str = "{},{},{}".format(int(time_step), "remove", node_id)
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
        xtime = []
        nnode = []
        time_step = 0
        # add initial nodes
        for i in range(self.desired_capacity):
            self.add_one_node(time_step)

        xtime.append(time_step)
        nnode.append(len(self.active_nodes))
        # loop through time steps (0 to total_time)
        # Adding interval from 0, til total_time hits
        while time_step < self.total_time:
            time_step += self.interval
            # add node
            if not self.no_adds:
                self.add_nodes(time_step)

            # remove node
            self.remove_node(time_step)

            xtime.append(time_step)
            nnode.append(len(self.active_nodes))

        return self.trace, np.array(xtime), np.array(nnode)



if __name__ == "__main__":
    # total_time = int(input("Enter the total time steps: "))

    parser = argparse.ArgumentParser(description='Generate trace for spot instances')
    parser.add_argument('--total_time', type=float, default=24, help='total time steps (hours)')
    parser.add_argument('--desired_capacity', type=int, default=30, help='desired capacity')
    parser.add_argument('--interval', type=float, default=1, help='interval (hours)')
    parser.add_argument('--seed', type=int, default=None, help='seed')
    parser.add_argument('--n_removals', type=int, default=None, help='numbers of removals per hour')
    parser.add_argument('--no_adds', action='store_true', help='no adds')
    args = parser.parse_args()

    total_time_ms = hour2ms(args.total_time)
    interval_ms = hour2ms(args.interval)

    trace_gen = TraceGenerator(total_time=total_time_ms, desired_capacity=args.desired_capacity, 
                               interval=interval_ms, seed=args.seed, n_removals=args.n_removals, no_adds=args.no_adds)
    # trace_gen = TraceGenerator(total_time)
    trace, xtime, nnode = trace_gen.generate_trace()

    hour = datetime.timedelta(hours=1)
    second = datetime.timedelta(seconds=1)
    millisecond = datetime.timedelta(milliseconds=1)
    milliseconds_per_second = second / millisecond
    milliseconds_per_hour = hour / millisecond
    # print(milliseconds_per_second)
    # time.sleep(10)

    for line in trace:
        print(line)

    plt.style.use('ggplot')
    # plot the trace
    plt.plot(xtime/milliseconds_per_hour, nnode, linewidth=2)
    plt.xlabel("Time (hours)")
    plt.ylabel("Number of nodes")
    plt.ylim([0, 32])
    plt.title("Spot instance trace")
    plt.tight_layout()
    plt.show()
