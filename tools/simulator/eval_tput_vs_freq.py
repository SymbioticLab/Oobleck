import os
import sys
import csv
import json
import heapq
import random
import logging
import argparse
import statistics
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

from utils import setup_logging, graph, generate_table
from bamboo_simulator import Result
from bamboo_simulator import Simulator as bamboo_simulator
from varuna_simulator import Simulator as varuna_simulator
from oobleck_simulator import Simulator as oobleck_simulator
from trace_generator import TraceGenerator

# Some global variables such as color and alpha are defined in utils.py
from utils import *

logger = logging.getLogger('simulation (bamboo, varun and oobleck)')

bar_trace_list = [
    "traces/6h.csv",
    "traces/3h.csv",
    "traces/1h.csv",
    "traces/30m.csv",
    "traces/10m.csv",
]

freqs = [
    "6h",
    "3h",
    "1h",
    "30m",
    "10m",
]

def merge_results(results: list):
    # given a list of Result objects, merge them into one Result object (average)
    try:
        removal_probability = statistics.mean([r.removal_probability for r in results])
    except:
        removal_probability = None
    preemption_mean = statistics.mean([r.preemption_mean for r in results])
    preemption_median = statistics.mean([r.preemption_median for r in results])
    preemption_stdev = statistics.mean([r.preemption_stdev for r in results])
    lifetime_mean = statistics.mean([r.lifetime_mean for r in results])
    lifetime_median = statistics.mean([r.lifetime_median for r in results])
    lifetime_stdev = statistics.mean([r.lifetime_stdev for r in results])
    num_preemptions = statistics.mean([r.num_preemptions for r in results])
    num_fatal_failures = statistics.mean([r.num_fatal_failures for r in results])
    num_steps_complete = statistics.mean([r.num_steps_complete for r in results])
    average_instances = statistics.mean([r.average_instances for r in results])
    average_performance = statistics.mean([r.average_performance for r in results])
    average_cost = statistics.mean([r.average_cost for r in results])
    average_value = statistics.mean([r.average_value for r in results])

    # return dict of merged results
    merge_result = {}
    merge_result['removal_probability'] = removal_probability
    merge_result['preemption_mean'] = preemption_mean
    merge_result['preemption_median'] = preemption_median
    merge_result['preemption_stdev'] = preemption_stdev
    merge_result['lifetime_mean'] = lifetime_mean
    merge_result['lifetime_median'] = lifetime_median
    merge_result['lifetime_stdev'] = lifetime_stdev
    merge_result['num_preemptions'] = num_preemptions
    merge_result['num_fatal_failures'] = num_fatal_failures
    merge_result['num_steps_complete'] = num_steps_complete
    merge_result['average_instances'] = average_instances
    merge_result['average_performance'] = average_performance
    merge_result['average_cost'] = average_cost
    merge_result['average_value'] = average_value
    return merge_result

def run_simulator_on_traces(model, bar_trace_list):
    bar_result_dict = {'bamboo':{}, 'varuna':{}, 'oobleck':{}}
    for trace_path in tqdm(bar_trace_list):
        csv_file = open(trace_path, 'r')
        simulator = bamboo_simulator(
            seed=seed,
            start_hour=0,
            generate_addition_probabilities=None,
            removal_probability=None,
            generate_graphs=False,
            spot_instance_trace=csv_file,
            model=model,
            steps_per_run=steps_per_run, 
            time_hard_stop=hard_stop
        )
        result = simulator.simulate()
        bar_result_dict['bamboo'][trace_path] = result.average_performance
        csv_file.close()

        csv_file = open(trace_path, 'r')
        simulator = varuna_simulator(
            seed=seed,
            start_hour=0,
            generate_addition_probabilities=None,
            removal_probability=None,
            generate_graphs=False,
            spot_instance_trace=csv_file,
            model=model,
            steps_per_run=steps_per_run, 
            time_hard_stop=hard_stop, 
            no_small_model_fallback=False
        )
        result = simulator.simulate()
        bar_result_dict['varuna'][trace_path] = result.average_performance
        csv_file.close()

        csv_file = open(trace_path, 'r')
        simulator = oobleck_simulator(
            seed=seed,
            start_hour=0,
            generate_addition_probabilities=None,
            removal_probability=None,
            generate_graphs=False,
            spot_instance_trace=csv_file,
            model=model,
            steps_per_run=steps_per_run, 
            time_hard_stop=hard_stop
        )
        result = simulator.simulate()
        bar_result_dict['oobleck'][trace_path] = result.average_performance
        csv_file.close()

    return bar_result_dict

def plot_results(result_dict, model_name, save_path=None, show=False):
    bamboo_oom = False
    if list(result_dict['bamboo'].values())[0]==0:
        bamboo_oom = True

    bar_results_data = {}
    for k in bar_trace_list:
        bar_results_data[k] = {}
        for kk, vv in result_dict.items():
            bar_results_data[k][kk] = vv[k]
    # set width of bars
    barWidth = 0.2
    # set positions of bars on x-axis
    X = np.arange(len(freqs))
    plt.figure(figsize=(7, 4))
    for i, (x, k) in enumerate(zip(X, bar_results_data.keys())):
        data_dict = bar_results_data[k]
        if bamboo_oom:
            plt.text(x-barWidth-0.05, 0.1, 'OOM', fontsize=12, color=BAMBOO_COLOR, weight='bold', rotation=90)
        if i==0:
            # get label one time
            plt.bar(x-barWidth, data_dict['bamboo'], color=BAMBOO_COLOR, width=barWidth, label="bamboo", alpha=ALPHA)
            plt.bar(x, data_dict['varuna'], color=VARUNA_COLOR, width=barWidth, label="varuna", alpha=ALPHA)
            plt.bar(x+barWidth, data_dict['oobleck'], color=OOBLECK_COLOR, width=barWidth, label="oobleck", alpha=ALPHA)
        else:
            plt.bar(x-barWidth, data_dict['bamboo'], color=BAMBOO_COLOR, width=barWidth, alpha=ALPHA)
            plt.bar(x, data_dict['varuna'], color=VARUNA_COLOR, width=barWidth, alpha=ALPHA)
            plt.bar(x+barWidth, data_dict['oobleck'], color=OOBLECK_COLOR, width=barWidth, alpha=ALPHA)

    plt.xticks(X, freqs)
    plt.ylabel("Throughput (sample / second)")
    plt.xlabel("Preemption Frequency")
    plt.tight_layout()
    plt.title(model_name, loc='left')
    # plt.legend(loc='upper left', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=False, shadow=False)
    plt.legend(loc='upper left', fontsize=10)
    if save_path is not None:
        plt.savefig(save_path)
        print("\nsave plot to {}".format(save_path))
    
    if show:
        fig = plt.gcf()
        fig.canvas.manager.set_window_title(model_name)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running simulation for bamboo, varuna and oobleck on traces with different preemption frequency")
    parser.add_argument("-m", "--model", required=True, metavar="", type=str, 
                        help="choose from BERT-large, GPT-2-ours, GPT-3-medium, GPT-3-2_7b, GPT-3-6_7b")
    parser.add_argument("-s", "--save_path", default=None, metavar="FILE", type=str, help="save plot to path")
    parser.add_argument("--show", action='store_true', help="show plot")
    parser.add_argument("--random_state", default=0, type=int, metavar='', help="random_state")
    args = parser.parse_args()

    model = args.model
    save_path = args.save_path
    random_state = args.random_state
    if model not in ['BERT-large', 'GPT-2-ours', 'GPT-3-medium', 'GPT-3-2_7b', 'GPT-3-6_7b']:
        raise ValueError("model should be one of BERT-large, GPT-2-ours, GPT-3-medium, GPT-3-2_7b, GPT-3-6_7b")
    
    bar_result_dict = run_simulator_on_traces(model, bar_trace_list)
    plot_results(bar_result_dict, model, save_path=save_path, show=args.show)
