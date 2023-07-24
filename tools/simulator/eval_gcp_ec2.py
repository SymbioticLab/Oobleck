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

# Some global variables such as color and alpha are defined in sim_utils.py
from utils import *

logger = logging.getLogger('simulation (bamboo, varun and oobleck)')

gcp_path = "traces/scaled/scale_gcp-trace.csv"
ec2_path = "traces/scaled/scale_p3-trace.csv"

# trace sacling
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

def get_time_gaps(trace, count_zero=True):
    remove_times = []
    gaps = []
    for event in trace:
        time_step, event_type, name = event
        if event_type == 'add':
            continue
        elif event_type == 'remove':
            remove_times.append(time_step)
    for i in range(len(remove_times) - 1):
        gap = remove_times[i+1] - remove_times[i]
        if gap!=0: gaps.append(gap)
        if gap==0 and count_zero: gaps.append(gap)
    return gaps

def ms_to_min(milliseconds):
    minutes = milliseconds / (1000 * 60)
    return minutes

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def run_simulator_on_traces(models, trace_path):
    all_results = {'bamboo':{}, 'varuna':{}, 'oobleck':{}}
    model_res_dict = {}
    for model in tqdm(models):
        all_results['bamboo'][model] = []
        all_results['varuna'][model] = []
        all_results['oobleck'][model] = []
        model_res_dict[model] = {}

        csv_file = open(trace_path, 'r')
        simulator = bamboo_simulator(
            seed=seed,
            start_hour=0,
            generate_addition_probabilities=None,
            removal_probability=None,
            generate_graphs=False,
            spot_instance_trace=csv_file,
            model=model,
            time_hard_stop=True, 
            detail_res=True
        )
        result, detail_res = simulator.simulate()
        model_res_dict[model]['bamboo'] = detail_res
        all_results['bamboo'][model] = result.average_performance
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
            steps_per_run=2000000000, 
            time_hard_stop=True, 
            detail_res=True, no_small_model_fallback=True
        )
        result, detail_res = simulator.simulate()
        model_res_dict[model]['varuna'] = detail_res
        all_results['varuna'][model] = result.average_performance
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
            steps_per_run=2000000000, 
            time_hard_stop=True, 
            detail_res=True
        )
        result, detail_res = simulator.simulate()
        model_res_dict[model]['oobleck'] = detail_res
        all_results['oobleck'][model] = result.average_performance
        csv_file.close()
    return model_res_dict

# ploting
title_size = 11
ylabel_size = 10
xlabel_size = 10 

def subplot_trace(ax, xlabel, ylabel, xs, ys, xmax, average, color='b', title=""):
    ymax = max(ys) + 1
    # plt.figure(figsize=(4.5, 3))
    ax.plot(xs, ys, linewidth=1.5, color=color)
    ax.set_xlabel(xlabel, fontsize=xlabel_size)
    ax.set_ylabel(ylabel, fontsize=ylabel_size)
    for scale in [1, 2, 6, 12, 24, 48, 72]:
        xticks = list(range(0, xmax + 1, scale))
        if len(xticks) < 7:
            break
    if xmax not in xticks:
        xticks.append(xmax)
    ax.set_xticks(xticks)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=title_size)
    ax.hlines(average, 0, xmax, color='tab:blue', linestyles='dotted')
    # ax.tight_layout()
    # return ax

def subplot_tput(ax, model_res_dict, model='BERT-large', show_ylabel=True, scaling_factor=30):
    tmp = {}
    tmp['varuna'] = model_res_dict[model]['varuna']
    tmp['bamboo'] = model_res_dict[model]['bamboo']
    tmp['oobleck'] = model_res_dict[model]['oobleck']
    plot_colors = {'bamboo': BAMBOO_COLOR, 'varuna': VARUNA_COLOR, 'oobleck': OOBLECK_COLOR}

    ymax = 0

    # plt.figure(figsize=(5.5, 3.5))
    duration_hours_whole = 0
    for system, v in tmp.items():
        if v['duration_hours_whole'] > duration_hours_whole:
            duration_hours_whole = v['duration_hours_whole']

        performance_xs, performance_ys, average_performance = v['performance']
        
        # tmp fix for varuna
        if len(performance_xs)!=len(performance_ys):
            performance_xs.append(performance_xs[-1])
            performance_xs.append(performance_xs[-1])
            
        # due to fallback varuna no progress
        if system=='varuna' and '6_7b' in model:
            performance_ys = [0.]*len(performance_xs)

        # find 6min mark (hour)
        window_size = scaling_factor / 60
        for i, v in enumerate(performance_xs):
            if v>=window_size:
                window_size_idx = i
                break
        xs = moving_average(performance_xs, window_size_idx)
        ys = moving_average(performance_ys, window_size_idx)

        ymax = max(ymax, max(ys))
        
        if system=='oobleck':
            ax.plot(xs, ys, linewidth=2, color=plot_colors[system], label=system, alpha=ALPHA)
        else:
            ax.plot(xs, ys, linewidth=1, color=plot_colors[system], label=system, alpha=ALPHA)
        ax.hlines(average_performance, 0, duration_hours_whole, color=plot_colors[system], linestyles='dotted', linewidth=2)
        # plt.hlines(np.mean(performance_ys), 0, duration_hours_whole, color=plot_colors[system], linestyles='dotted', linewidth=2)

    ax.set_title(model, fontsize=title_size)
    if show_ylabel:
        ax.set_ylabel("Throughput (sample / second)", fontsize=ylabel_size)
    ax.set_xlabel("Time (hours)", fontsize=xlabel_size)
    for scale in [1, 2, 6, 12, 24, 48, 72]:
        xticks = list(range(0, duration_hours_whole + 1, scale))
        if len(xticks) < 7:
            break
    if duration_hours_whole not in xticks:
        xticks.append(duration_hours_whole)
    ax.set_xticks(xticks)
    ax.set_xlim(0, duration_hours_whole)
    ax.set_ylim(0, ymax+0.1*ymax)

    # handles, labels = plt.gca().get_legend_handles_labels()
    handles, labels = ax.get_legend_handles_labels()
    # print(handles)
    order = [2,0,1]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
               loc='upper center', ncol=3, fancybox=False, shadow=False, fontsize=8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running simulation for bamboo, varuna and oobleck on the GCP and EC3 traces")
    # parser.add_argument("-s", "--save_path", default=None, metavar="FILE", type=str, help="save plot to path")
    parser.add_argument("--show", action='store_true', help="show plot")
    args = parser.parse_args()

    models = ['BERT-large', 'GPT-2-ours', 'GPT-3-medium', 'GPT-3-2_7b', 'GPT-3-6_7b']

    # Run on GCP trace
    model_res_dict = run_simulator_on_traces(models, gcp_path)
    detail_res = model_res_dict[models[0]]['oobleck']

    moving_avg_factor = {'BERT-large':18, 'GPT-2-ours':25, 'GPT-3-medium':25, 'GPT-3-2_7b':60, 'GPT-3-6_7b':33}
    fig, axs = plt.subplots(2, 3, figsize=(4.5*3, 3*2))

    subplot_trace(axs[0][0], xlabel='Time (hours)', ylabel='# Instances', 
            xs=detail_res['instances'][0][:-1], ys=detail_res['instances'][1][:-1], 
            xmax=detail_res["duration_hours_whole"], average=detail_res['instances'][2], title="GCP a2-highgpu-1g instances")

    # plot tput for 5 model, fill up the subplots from left to right, top to bottom
    subplot_tput(axs[0][1], model_res_dict, model=models[0], show_ylabel=True, scaling_factor=moving_avg_factor[models[0]])
    subplot_tput(axs[0][2], model_res_dict, model=models[1], show_ylabel=False, scaling_factor=moving_avg_factor[models[1]])
    subplot_tput(axs[1][0], model_res_dict, model=models[2], show_ylabel=True, scaling_factor=moving_avg_factor[models[2]])
    subplot_tput(axs[1][1], model_res_dict, model=models[3], show_ylabel=False, scaling_factor=moving_avg_factor[models[3]])
    subplot_tput(axs[1][2], model_res_dict, model=models[4], show_ylabel=False, scaling_factor=moving_avg_factor[models[4]])

    fig.tight_layout()
    gcp_save_path = "gcp_trace_eval.pdf"
    plt.savefig(gcp_save_path)

    if args.show:
        fig = plt.gcf()
        fig.canvas.manager.set_window_title('GCP Trace Evaluation')
        plt.show()

    # Run on EC2 trace
    model_res_dict = run_simulator_on_traces(models, ec2_path)
    detail_res = model_res_dict[models[0]]['oobleck']

    moving_avg_factor = {'BERT-large':20, 'GPT-2-ours':5, 'GPT-3-medium':5, 'GPT-3-2_7b':20, 'GPT-3-6_7b':11}
    fig, axs = plt.subplots(2, 3, figsize=(4.5*3, 3*2))

    subplot_trace(axs[0][0], xlabel='Time (hours)', ylabel='# Instances', 
            xs=detail_res['instances'][0][:-1], ys=detail_res['instances'][1][:-1], 
            xmax=detail_res["duration_hours_whole"], average=detail_res['instances'][2], title="EC2 P3 instances")

    # plot tput for 5 model, fill up the subplots from left to right, top to bottom
    subplot_tput(axs[0][1], model_res_dict, model=models[0], show_ylabel=True, scaling_factor=moving_avg_factor[models[0]])
    subplot_tput(axs[0][2], model_res_dict, model=models[1], show_ylabel=False, scaling_factor=moving_avg_factor[models[1]])
    subplot_tput(axs[1][0], model_res_dict, model=models[2], show_ylabel=True, scaling_factor=moving_avg_factor[models[2]])
    subplot_tput(axs[1][1], model_res_dict, model=models[3], show_ylabel=False, scaling_factor=moving_avg_factor[models[3]])
    subplot_tput(axs[1][2], model_res_dict, model=models[4], show_ylabel=False, scaling_factor=moving_avg_factor[models[4]])

    fig.tight_layout()
    ec2_save_path = "ec2_trace_eval.pdf"
    plt.savefig(ec2_save_path)

    print()
    print("save plot to {}".format(gcp_save_path))
    print("save plot to {}".format(ec2_save_path))

    if args.show:
        fig = plt.gcf()
        fig.canvas.manager.set_window_title('EC2 P3 Trace Evaluation')
        plt.show()