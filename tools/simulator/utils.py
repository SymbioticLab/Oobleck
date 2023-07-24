import argparse
import logging
import os
import time
import multiprocessing

from colorama import Fore, Style

import csv
import sys
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# adjust color here
BAMBOO_COLOR= '#008000'
VARUNA_COLOR = '#000000'
OOBLECK_COLOR = '#FF8A1D'
VARUNA_NOCKPT_COLOR = '#aaaaaa'
ALPHA = 1

seed = None
hard_stop = True
steps_per_run = 200000

# Set the default text font size
# plt.rc('font', size=16)
# Set the axes title font size
plt.rc('axes', titlesize=15)
# Set the axes labels font size
plt.rc('axes', labelsize=15)
# Set the font size for x tick labels
plt.rc('xtick', labelsize=12)
# Set the font size for y tick labels
plt.rc('ytick', labelsize=12)
# Set the legend font size
plt.rc('legend', fontsize=13)
# Set the font size of the figure title
# plt.rc('figure', titlesize=20)

def compute_tput(num_sample, time_used):
    return num_sample / time_used

def tput2iter_time_ms(tput, batch_size):
    # tput sample / second
    # batch_size num samples
    # return time in ms
    return (batch_size / tput) * 1000

def read_trace(trace_path):
    # trace = [(delta, event, name)]
    trace = []
    with open(trace_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            delta, event, name = row
            delta = int(delta)
            trace.append( (delta, event, name) )
    return trace

def trace2n_node(trace):
    n_nodes_time_dict = {}
    
    tmp_num = 0
    for event in trace:
        t, e, name = event
        if e=='add':
            tmp_num += 1
        elif e=='remove':
            tmp_num -= 1
        n_nodes_time_dict[t] = tmp_num
    return list(n_nodes_time_dict.keys()), list(n_nodes_time_dict.values())

def n_node2trace(xtimes, n_nodes):
    import random 
    trace = []
    node_counter = 1
    activate_nodes = []
    for i, (t, n) in enumerate(zip(xtimes, n_nodes)):
        
        if len(activate_nodes) == n:
            trace.append("{},add,node{}".format(t, node_counter))
            activate_nodes.append("node{}".format(node_counter))
            node_counter+=1
            random_element = random.choice(activate_nodes)
            trace.append("{},remove,{}".format(t, random_element))
            activate_nodes.remove(random_element)
        # need add
        elif len(activate_nodes) < n:
            while len(activate_nodes)<n:
                trace.append("{},add,node{}".format(t, node_counter))
                activate_nodes.append("node{}".format(node_counter))
                node_counter+=1
        # need remove
        elif len(activate_nodes) > n:
            while len(activate_nodes) > n:
                random_element = random.choice(activate_nodes)
                trace.append("{},remove,{}".format(int(t), random_element))
                activate_nodes.remove(random_element)
    return trace

def plot_trace(xlabel, ylabel, xs, ys, xmax, average, color='b', title=""):
    ymax = max(ys) + 1
    plt.figure(figsize=(4.5, 3))
    plt.plot(xs, ys, linewidth=0.8, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for scale in [1, 2, 6, 12, 24, 48, 72]:
        xticks = list(range(0, xmax + 1, scale))
        if len(xticks) < 7:
            break
    if xmax not in xticks:
        xticks.append(xmax)
    plt.xticks(xticks)
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.title(title)
    plt.hlines(average, 0, xmax, color='tab:blue', linestyles='dotted')
    plt.tight_layout()
    # plt.savefig("p3_res/p3_trace.pdf", bbox_inches='tight')
    plt.show()


class ProjectPactumFormatter(logging.Formatter):

	def __init__(self):
		self.created = time.time()

	def format(self, record):
		reltime = record.created - self.created
		COLORS = {
			logging.DEBUG: 35,
			logging.INFO: 36,
			logging.WARNING: 33,
			logging.ERROR: 31,
		}
		fmt = '\x1B[1;{color}m[{reltime:.3f} p%(process)d/t%(thread)d %(levelname)s %(name)s]\x1B[m \x1B[{color}m%(message)s\x1B[m'
		formatter = logging.Formatter(fmt.format(color=COLORS[record.levelno], reltime=reltime))
		return formatter.format(record)

def parse(args):
	parser = argparse.ArgumentParser(prog='Oobleck',
	                                 description='Oobleck')

	parser.add_argument(
		'--version', action='version',
		version=f'{Fore.BLUE}{Style.BRIGHT}Oobleck{Style.RESET_ALL}'
		        f' {Style.BRIGHT}{Style.RESET_ALL}')

	return parser.parse_args(args)

def setup_logging():
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(ProjectPactumFormatter())
	logging.basicConfig(level=logging.DEBUG, handlers=[stream_handler])

	logging.getLogger('botocore.auth').setLevel(logging.WARNING)
	logging.getLogger('botocore.client').setLevel(logging.WARNING)
	logging.getLogger('botocore.credentials').setLevel(logging.WARNING)
	logging.getLogger('botocore.endpoint').setLevel(logging.WARNING)
	logging.getLogger('botocore.handlers').setLevel(logging.WARNING)
	logging.getLogger('botocore.hooks').setLevel(logging.WARNING)
	logging.getLogger('botocore.httpsession').setLevel(logging.WARNING)
	logging.getLogger('botocore.loaders').setLevel(logging.WARNING)
	logging.getLogger('botocore.parsers').setLevel(logging.WARNING)
	logging.getLogger('botocore.retryhandler').setLevel(logging.WARNING)
	logging.getLogger('botocore.utils').setLevel(logging.WARNING)
	logging.getLogger('boto3.resources.action').setLevel(logging.WARNING)
	logging.getLogger('boto3.resources.collection').setLevel(logging.WARNING)
	logging.getLogger('boto3.resources.factory').setLevel(logging.WARNING)
	logging.getLogger('boto3.resources.model').setLevel(logging.WARNING)

	logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)

	logging.getLogger('matplotlib').setLevel(logging.WARNING)


def graph(xlabel, xs, xmax, ylabel, ys, ymax, average,
          on_demand=None, out=None, show=False):
        import matplotlib.pyplot as plt

        # sizes: xx-small, x-small, small, medium, large, x-large, xx-large
        params = {
            # 'font.family': 'Inter',
            'font.family': 'sans-serif',
            'legend.fontsize': 'x-small',
            'axes.labelsize': 'x-small',
            'axes.titlesize': 'x-small',
            'xtick.labelsize': 'x-small',
            'ytick.labelsize': 'x-small',
            'figure.figsize': (3.0, 1.7),
        }
        plt.rcParams.update(params)

        plt.clf()

        plt.plot(xs, ys, linewidth=0.5)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        for scale in [1, 2, 6, 12, 24, 48, 72]:
            xticks = list(range(0, xmax + 1, scale))
            if len(xticks) < 7:
                break
        if xmax not in xticks:
            xticks.append(xmax)
        plt.xticks(xticks)

        plt.xlim(0, xmax)
        plt.ylim(0, ymax)

        plt.hlines(average, 0, xmax, color='tab:blue', linestyles='dotted')
        # if on_demand is not None:
        #     plt.hlines(on_demand, 0, xmax, color='tab:red', linestyles='dashed')

        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if out is not None:
           plt.savefig(
               out,
               bbox_inches='tight',
               pad_inches=0
           )

        if show:
            plt.show()
            
def generate_table():
    logging.getLogger('oobleck.tools.simulator').setLevel(logging.WARNING)

    count = 0

    removal_probabilities = [0.01, 0.05, 0.10, 0.25, 0.50]
    all_preemptions = {}
    all_interruptions = {}
    all_lifetimes = {}
    all_fatal_failures = {}
    all_instances = {}
    all_performance = {}
    all_cost = {}
    all_value = {}
    for removal_probability in removal_probabilities:
        all_preemptions[removal_probability] = []
        all_interruptions[removal_probability] = []
        all_lifetimes[removal_probability] = []
        all_fatal_failures[removal_probability] = []
        all_instances[removal_probability] = []
        all_performance[removal_probability] = []
        all_cost[removal_probability] = []
        all_value[removal_probability] = []

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        simulations = []
        for removal_probability in removal_probabilities:
            for seed in range(1, 10_001):
                simulations.append((removal_probability, seed))

        for result in pool.imap_unordered(simulate, simulations):
            removal_probability = result.removal_probability
            all_preemptions[removal_probability].append(result.num_preemptions)
            all_interruptions[removal_probability].append(result.preemption_mean)
            all_lifetimes[removal_probability].append(result.lifetime_mean)
            all_fatal_failures[removal_probability].append(result.num_fatal_failures)
            all_instances[removal_probability].append(result.average_instances)
            all_performance[removal_probability].append(result.average_performance)
            all_cost[removal_probability].append(result.average_cost)
            all_value[removal_probability].append(result.average_value)

            count += 1
            if count % 100 == 0:
                logger.info(f'{count} simulations complete')

    print('Probability', 'Preemptions', 'Interruptions', 'Lifetimes', 'Fatal Failures', 'Instances', 'Performance', '     Cost', '    Value',
          sep=' & ', end=' \\\\\n')
    for removal_probability in removal_probabilities:
        print(f'{removal_probability:11.2f}',
            '{:11.2f}'.format(statistics.mean(all_preemptions[removal_probability])),
            '{:13.2f}'.format(statistics.mean(all_interruptions[removal_probability])),
            '{:9.2f}'.format(statistics.mean(all_lifetimes[removal_probability])),
            '{:14.2f}'.format(statistics.mean(all_fatal_failures[removal_probability])),
            '{:9.2f}'.format(statistics.mean(all_instances[removal_probability])),
            '{:11.2f}'.format(statistics.mean(all_performance[removal_probability])),
            '{:9.2f}'.format(statistics.mean(all_cost[removal_probability])),
            '{:9.2f}'.format(statistics.mean(all_value[removal_probability])),
            sep=' & ', end=' \\\\\n'
        )
