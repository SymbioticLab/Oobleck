import csv
import dataclasses
import datetime
import enum
import heapq
import logging
import math
import random
import statistics
import typing

from pprint import pprint
from time import sleep

logger = logging.getLogger(__name__)


def tput2iter_time_ms(tput, batch_size):
    # tput sample / second
    # batch_size num samples
    # return time in ms
    return int((batch_size / tput) * 1000)


class EventKind(enum.IntEnum):
    SPOT_INSTANCE_ADD = 1
    SPOT_INSTANCE_REMOVE = 2
    SPOT_INSTANCE_GENERATE = 3
    SPOT_INSTANCE_READY = 4
    GLOBAL_RENDEZVOUS_TIMEOUT = 5
    LOCAL_RENDEZVOUS_TIMEOUT = 6
    TRAINING_STEP_COMPLETE = 7
    CHECKPOINT_COMPLETE = 8


@dataclasses.dataclass(order=True)
class Event:
    delta: int
    kind: EventKind
    data: typing.Dict = dataclasses.field(compare=False)


class SystemStatus(enum.Enum):
    STOPPED = enum.auto()
    RENDEZVOUS = enum.auto()
    RUNNING = enum.auto()


class NodeStatus(enum.Enum):
    CREATING = enum.auto()
    READY = enum.auto()
    RUNNING = enum.auto()


@dataclasses.dataclass
class Result:
    removal_probability: float
    preemption_mean: float
    preemption_median: float
    preemption_stdev: float
    lifetime_mean: float
    lifetime_median: float
    lifetime_stdev: float
    num_preemptions: int
    num_fatal_failures: int
    num_steps_complete: int
    average_instances: float
    average_performance: float
    average_cost: float
    average_value: float


class SpotInstance:
    def __init__(self, name, start):
        self.name = name
        self.start = start
        self.status = NodeStatus.CREATING
        self.global_id = None
        self.active_coordinates = []
        self.previous_coordinates = []

    def is_creating(self):
        return self.status == NodeStatus.CREATING

    def set_ready(self):
        self.status = NodeStatus.READY

    def is_ready(self):
        return self.status == NodeStatus.READY

    def set_running(self):
        self.status = NodeStatus.RUNNING

    def is_running(self):
        return self.status == NodeStatus.RUNNING

    def uptime(self, end):
        return end - self.start

    def __str__(self):
        return self.name


class Simulator:
    def __init__(
        self,
        seed=None,
        start_hour=None,
        model="BERT-large",
        spot_instance_trace=None,
        generate_addition_probabilities=False,
        removal_probability=None,
        generate_graphs=False,
        steps_per_run=None,
        time_hard_stop=False,
        detail_res=False,
        no_ckpt_overhead=False,
        checkpoint_per_iter=10,
        batch_size=None,
        no_small_model_fallback=False,
        no_restart_overhead=False,
    ):
        self.spot_instance_trace = spot_instance_trace
        self.generate_graphs = generate_graphs
        self.steps_per_run = steps_per_run
        self.time_hard_stop = time_hard_stop
        self.detail_res = detail_res
        self.no_ckpt_overhead = no_ckpt_overhead
        self.checkpoint_per_iter = checkpoint_per_iter
        self.batch_size = batch_size
        self.no_small_model_fallback = no_small_model_fallback
        self.no_restart_overhead = no_restart_overhead

        self.effective_time = 0
        self.checkpoint_time = 0
        self.restart_time = 0
        self.fallback_loss_time = 0

        logger.info(f"Simulating: varuna")
        # sleep(3)

        self.seed = seed
        if self.seed is not None:
            self.r = random.Random(self.seed)
            logger.info(f"Using seed: {self.seed}")
        else:
            self.r = random.Random()
        self.generate_addition_probabilities = generate_addition_probabilities
        self.removal_probability = removal_probability

        self.hour = datetime.timedelta(hours=1)
        self.second = datetime.timedelta(seconds=1)
        self.millisecond = datetime.timedelta(milliseconds=1)
        self.milliseconds_per_second = self.second / self.millisecond
        self.milliseconds_per_hour = self.hour / self.millisecond

        self.spot_instance_name_format = "node{id}"
        self.spot_instance_next_id = 1
        if not generate_addition_probabilities:
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
        if removal_probability is None:
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
            self.spot_instance_removal_probability = {}
            for hour in range(24):
                self.spot_instance_removal_probability[hour] = removal_probability

        self.start_hour = start_hour
        self.spot_instance_creation_time = 45_000  # milliseconds

        self.global_rendezvous_timeout_delta = 10000

        self.spot_instances = {}
        self.rendezvous_version = 0
        self.rendezvous = []
        self.num_workers_waiting = 0
        self.num_pipelines = 0
        self.num_stages = 0

        self.num_steps_complete = 0
        self.num_fatal_failures = 0
        self.num_spot_instance_removals = 0

        self.fallback_slowdown = 1
        self.fallback_event = None
        self.fallback_handled = False

        self.spot_instance_removal_times = []
        self.spot_instance_lifetimes = []

        self.start_delta = None
        self.previous_step_complete_delta = 0

        self.events = []
        heapq.heapify(self.events)

        self.status = SystemStatus.STOPPED

        self.on_demand_cost_per_hour = 3.06
        self.spot_instance_cost_per_hour = 0.91

        self.should_reconfigure_counter = 0
        self.rendezvous_start_counter = 0
        # self.checkpointing_failure = False
        # self.checkpoint_per_iter = 10
        self.prev_checkpoint_delta = -1

        self.no_fallback = False
        self.node_will_be_removed = False
        self.sum_gpu_overhead = 0.

        # NEED update, faster then bamboo
        if model == "BERT-large":
            self.samples_per_step = 8192
            if self.no_small_model_fallback:
                self.no_fallback = True

            if self.steps_per_run is None:
                self.steps_per_run = 30_000

            self.spot_instance_desired_capacity = 15
            self.num_stages_target = 12  # NO USE HERE
            self.simulate_step_delta = self.bertLarge_simulate_step_delta
            # self.local_rendezvous_timeout_delta = 76000
            self.local_rendezvous_timeout_delta = 76_000
            self.checkpointing_overhead = int(15.8 * 1000)
            # 15min to ms: 600000
            # self.checkpointing_overhead = 650000

            self.on_demand_num_instances = 4 * 5
            self.on_demand_cost = (
                self.on_demand_num_instances * self.on_demand_cost_per_hour
            )
            self.on_demand_performance = self.samples_per_step / 2.51
            self.on_demand_value = self.on_demand_performance / self.on_demand_cost
        elif model == "GPT-2-ours":
            self.samples_per_step = 8192 // 8
            if self.no_small_model_fallback:
                self.no_fallback = True

            if self.steps_per_run is None:
                self.steps_per_run = 30_000
            self.spot_instance_desired_capacity = 15
            self.simulate_step_delta = self.gpt_2_ours_simulate_step_delta
            self.local_rendezvous_timeout_delta = 83_000
            self.checkpointing_overhead = int(11.5 * 1000)

            self.num_stages_target = 12
            self.on_demand_num_instances = 4 * 8
            self.on_demand_cost = (
                self.on_demand_num_instances * self.on_demand_cost_per_hour
            )
            self.on_demand_performance = self.samples_per_step / 3.37
            self.on_demand_value = self.on_demand_performance / self.on_demand_cost
        elif model == "GPT-3-medium":
            if self.no_small_model_fallback:
                self.no_fallback = True
            self.samples_per_step = 8192 // 10
            if self.steps_per_run is None:
                self.steps_per_run = 60_000
            self.spot_instance_desired_capacity = 15
            self.simulate_step_delta = self.gpt_3m_simulate_step_delta
            self.local_rendezvous_timeout_delta = 78_700
            self.checkpointing_overhead = 11_200

            self.num_stages_target = 12
            self.on_demand_num_instances = 4 * 8
            self.on_demand_cost = (
                self.on_demand_num_instances * self.on_demand_cost_per_hour
            )
            self.on_demand_performance = self.samples_per_step / 3.37
            self.on_demand_value = self.on_demand_performance / self.on_demand_cost
        elif model == "GPT-3-2_7b":
            self.samples_per_step = 1024 / 2
            if self.steps_per_run is None:
                self.steps_per_run = 30_000
            self.spot_instance_desired_capacity = 15
            self.simulate_step_delta = self.gpt_3_2_7b_simulate_step_delta
            self.local_rendezvous_timeout_delta = 515_000
            self.checkpointing_overhead = 50_000

            self.num_stages_target = 12
            self.on_demand_num_instances = 4 * 8
            self.on_demand_cost = (
                self.on_demand_num_instances * self.on_demand_cost_per_hour
            )
            self.on_demand_performance = self.samples_per_step / 3.37
            self.on_demand_value = self.on_demand_performance / self.on_demand_cost
        elif model == "GPT-3-6_7b":
            self.samples_per_step = 1024
            if self.steps_per_run is None:
                self.steps_per_run = 30_000
            self.spot_instance_desired_capacity = 15
            self.simulate_step_delta = self.gpt_3_6_7b_simulate_step_delta
            self.local_rendezvous_timeout_delta = 701_000
            self.checkpointing_overhead = 120_000
            
            # GOD speed
            # self.checkpointing_overhead = 0
            # self.local_rendezvous_timeout_delta = 0
            # self.global_rendezvous_timeout_delta = 0

            self.num_stages_target = 12
            self.on_demand_num_instances = 4 * 8
            self.on_demand_cost = (
                self.on_demand_num_instances * self.on_demand_cost_per_hour
            )
            self.on_demand_performance = self.samples_per_step / 3.37
            self.on_demand_value = self.on_demand_performance / self.on_demand_cost

        ### ALL NN below NOT USE!
        elif model == "GPT-2":
            self.samples_per_step = 32
            self.steps_per_run = 188_828

            self.spot_instance_desired_capacity = 48
            self.simulate_step_delta = self.gpt_2_simulate_step_delta
            self.local_rendezvous_timeout_delta = 20_000  # milliseconds
            self.num_stages_target = 12

            self.on_demand_num_instances = 4 * 8
            self.on_demand_cost = (
                self.on_demand_num_instances * self.on_demand_cost_per_hour
            )
            self.on_demand_performance = self.samples_per_step / 3.37
            self.on_demand_value = self.on_demand_performance / self.on_demand_cost
        elif model == "BERT":
            self.samples_per_step = 264
            self.steps_per_run = 22_194

            self.spot_instance_desired_capacity = 48
            self.num_stages_target = 12
            self.simulate_step_delta = self.bert_simulate_step_delta
            self.local_rendezvous_timeout_delta = 20_000  # milliseconds

            self.on_demand_num_instances = 4 * 5
            self.on_demand_cost = (
                self.on_demand_num_instances * self.on_demand_cost_per_hour
            )
            self.on_demand_performance = self.samples_per_step / 2.51
            self.on_demand_value = self.on_demand_performance / self.on_demand_cost
        elif model == "ResNet":
            self.samples_per_step = 384

            self.spot_instance_desired_capacity = 32
            self.simulate_step_delta = self.resnet_simulate_step_delta
            self.local_rendezvous_timeout_delta = 10_000  # milliseconds
            self.num_stages_target = 8

            self.on_demand_num_instances = 4 * 5
            self.on_demand_cost = (
                self.on_demand_num_instances * self.on_demand_cost_per_hour
            )
            self.on_demand_performance = self.samples_per_step / 1.3
            self.on_demand_value = self.on_demand_performance / self.on_demand_cost
        elif model == "GNMT":
            self.samples_per_step = 288

            self.spot_instance_desired_capacity = 24
            self.simulate_step_delta = self.gnmt_simulate_step_delta
            self.local_rendezvous_timeout_delta = 10_000  # milliseconds
            self.num_stages_target = 6

            self.on_demand_num_instances = 4 * 4
            self.on_demand_cost = (
                self.on_demand_num_instances * self.on_demand_cost_per_hour
            )
            self.on_demand_performance = self.samples_per_step / 0.9
            self.on_demand_value = self.on_demand_performance / self.on_demand_cost
        elif model == "VGG":
            self.samples_per_step = 384

            self.spot_instance_desired_capacity = 24
            self.simulate_step_delta = self.vgg_simulate_step_delta
            self.local_rendezvous_timeout_delta = 5_000  # milliseconds
            self.num_stages_target = 6

            self.on_demand_num_instances = 4 * 4
            self.on_demand_cost = (
                self.on_demand_num_instances * self.on_demand_cost_per_hour
            )
            self.on_demand_performance = self.samples_per_step / 2.2
            self.on_demand_value = self.on_demand_performance / self.on_demand_cost
        elif model == "AlexNet":
            self.samples_per_step = 384

            self.spot_instance_desired_capacity = 24
            self.simulate_step_delta = self.alexnet_simulate_step_delta
            self.local_rendezvous_timeout_delta = 5_000  # milliseconds
            self.num_stages_target = 6

            self.on_demand_num_instances = 4 * 4
            self.on_demand_cost = (
                self.on_demand_num_instances * self.on_demand_cost_per_hour
            )
            self.on_demand_performance = self.samples_per_step / 1.6
            self.on_demand_value = self.on_demand_performance / self.on_demand_cost
        else:
            raise NotImplementedError
        self.model = model

    def generate_probabilities(self):
        probability = {}
        for hour in range(24):
            probability[hour] = self.r.random()
        return probability

    def bertLarge_simulate_step_delta(self):
        # set up self.step_delta base on number of active instances
        self.curr_gpu_overhead = 0.

        batch_size = self.samples_per_step
        num_active_instances = 0
        for name, instance in self.spot_instances.items():
            if instance.is_running() or instance.is_ready():
                num_active_instances += 1
        # milliseconds
        tputs = [
            343.1328746,
            331.6951121,
            329.5800552,
            326.5438348,
            306.0386227,
            298.3794267,
            290.038974,
            280.0744467,
            267.8972968,
            255.720147,
            243.5429971,
            232.9541711,
            222.3653452,
            211.7765192,
            202.5688445,
            193.3611697,
        ]
        tput_dict = {}
        for i in range(len(tputs)):
            tput_dict[30 - i] = tputs[i]
                
        if num_active_instances >= 30:
            self.local_rendezvous_timeout_delta = int(71.3 * 1000)
            self.step_delta = tput2iter_time_ms(tput_dict[30], batch_size)
        elif num_active_instances == 29:
            self.local_rendezvous_timeout_delta = int(67.6 * 1000)
            self.step_delta = tput2iter_time_ms(tput_dict[29], batch_size)
        elif num_active_instances == 28:
            self.local_rendezvous_timeout_delta = int(67.6 * 1000)
            self.step_delta = tput2iter_time_ms(tput_dict[28], batch_size)
        elif num_active_instances == 27:
            self.local_rendezvous_timeout_delta = int(49.1 * 1000)
            self.step_delta = tput2iter_time_ms(tput_dict[27], batch_size)
        elif num_active_instances == 26:
            self.local_rendezvous_timeout_delta = int(112 * 1000)
            self.step_delta = tput2iter_time_ms(tput_dict[26], batch_size)
        elif num_active_instances == 25:
            self.local_rendezvous_timeout_delta = int(108.3 * 1000)
            self.step_delta = tput2iter_time_ms(tput_dict[25], batch_size)
        elif num_active_instances == 24:
            self.local_rendezvous_timeout_delta = int(104.6 * 1000)
            self.step_delta = tput2iter_time_ms(tput_dict[24], batch_size)
        elif num_active_instances == 23:
            self.local_rendezvous_timeout_delta = int(100.9 * 1000)
            self.step_delta = tput2iter_time_ms(tput_dict[23], batch_size)
        elif num_active_instances == 22:
            self.local_rendezvous_timeout_delta = int(97.2 * 1000)
            self.step_delta = tput2iter_time_ms(tput_dict[22], batch_size)
        elif num_active_instances == 21:
            self.local_rendezvous_timeout_delta = int(93.5 * 1000)
            self.step_delta = tput2iter_time_ms(tput_dict[21], batch_size)
        elif num_active_instances == 20:
            self.local_rendezvous_timeout_delta = int(89.8 * 1000)
            self.step_delta = tput2iter_time_ms(tput_dict[20], batch_size)
        elif num_active_instances == 19:
            self.local_rendezvous_timeout_delta = int(86.1 * 1000)
            self.step_delta = tput2iter_time_ms(tput_dict[19], batch_size)
        elif num_active_instances == 18:
            self.local_rendezvous_timeout_delta = int(82.4 * 1000)
            self.step_delta = tput2iter_time_ms(tput_dict[18], batch_size)
        elif num_active_instances == 17:
            self.local_rendezvous_timeout_delta = int(78.7 * 1000)
            self.step_delta = tput2iter_time_ms(tput_dict[17], batch_size)
        elif num_active_instances == 16:
            self.local_rendezvous_timeout_delta = int(75 * 1000)
            self.step_delta = tput2iter_time_ms(tput_dict[16], batch_size)
        elif num_active_instances <= 15:
            self.local_rendezvous_timeout_delta = int(71.3 * 1000)
            self.step_delta = tput2iter_time_ms(tput_dict[15], batch_size)
        else:
            raise NotImplementedError

    def gpt_2_ours_simulate_step_delta(self):
        # set up self.step_delta base on number of active instances
        batch_size = self.samples_per_step
        num_active_instances = 0
        for name, instance in self.spot_instances.items():
            if instance.is_running() or instance.is_ready():
                num_active_instances += 1
        
        idle_gpu_overhead = {
                    30: 0,
                    29: 1,
                    28: 0,
                    27: 0,
                    26: 2,
                    25: 1,
                    24: 0,
                    23: 2,
                    22: 1,
                    21: 0,
                    20: 0,
                    19: 1,
                    18: 0,
                    17: 1,
                    16: 0,
                    15: 0
                }

        if num_active_instances in idle_gpu_overhead.keys():
            self.curr_gpu_overhead = idle_gpu_overhead[num_active_instances]
        else:
            self.curr_gpu_overhead = 0.
        
        # milliseconds
        tputs = [
            125.7855818,
            117.6280102,
            117.6280102,
            106.8250673,
            110.9162094,
            102.0772415,
            102.0772415,
            96.65353837,
            96.65353837,
            85.83389481,
            88.31094283,
            78.31133417,
            78.31133417,
            70.46360441,
            70.46360441,
            64.29935764,
        ]
        tput_dict = {}
        for i in range(len(tputs)):
            tput_dict[30 - i] = tputs[i]

        if num_active_instances >= 30:
            self.step_delta = tput2iter_time_ms(tput_dict[30], batch_size)
        elif num_active_instances == 29:
            self.step_delta = tput2iter_time_ms(tput_dict[29], batch_size)
        elif num_active_instances == 28:
            self.step_delta = tput2iter_time_ms(tput_dict[28], batch_size)
        elif num_active_instances == 27:
            self.step_delta = tput2iter_time_ms(tput_dict[27], batch_size)
        elif num_active_instances == 26:
            self.step_delta = tput2iter_time_ms(tput_dict[26], batch_size)
        elif num_active_instances == 25:
            self.step_delta = tput2iter_time_ms(tput_dict[25], batch_size)
        elif num_active_instances == 24:
            self.step_delta = tput2iter_time_ms(tput_dict[24], batch_size)
        elif num_active_instances == 23:
            self.step_delta = tput2iter_time_ms(tput_dict[23], batch_size)
        elif num_active_instances == 22:
            self.step_delta = tput2iter_time_ms(tput_dict[22], batch_size)
        elif num_active_instances == 21:
            self.step_delta = tput2iter_time_ms(tput_dict[21], batch_size)
        elif num_active_instances == 20:
            self.step_delta = tput2iter_time_ms(tput_dict[20], batch_size)
        elif num_active_instances == 19:
            self.step_delta = tput2iter_time_ms(tput_dict[19], batch_size)
        elif num_active_instances == 18:
            self.step_delta = tput2iter_time_ms(tput_dict[18], batch_size)
        elif num_active_instances == 17:
            self.step_delta = tput2iter_time_ms(tput_dict[17], batch_size)
        elif num_active_instances == 16:
            self.step_delta = tput2iter_time_ms(tput_dict[16], batch_size)
        elif num_active_instances <= 15:
            self.step_delta = tput2iter_time_ms(tput_dict[15], batch_size)
        else:
            raise NotImplementedError

    def gpt_3m_simulate_step_delta(self):
        # set up self.step_delta base on number of active instances
        batch_size = self.samples_per_step # should match the batch size above
        num_active_instances = 0
        for name, instance in self.spot_instances.items():
            if instance.is_running() or instance.is_ready():
                num_active_instances += 1
        
        idle_gpu_overhead = {
                    30: 0,
                    29: 1,
                    28: 0,
                    27: 0,
                    26: 2,
                    25: 1,
                    24: 0,
                    23: 2,
                    22: 1,
                    21: 0,
                    20: 0,
                    19: 1,
                    18: 0,
                    17: 1,
                    16: 0,
                    15: 0
                }
        if num_active_instances in idle_gpu_overhead.keys():
            self.curr_gpu_overhead = idle_gpu_overhead[num_active_instances]
        else:
            self.curr_gpu_overhead = 0.

        # milliseconds
        tputs = [
            41.99773336,
            38.88588843,
            38.88588843,
            35.85035677,
            33.42576818,
            33.42576818,
            33.42576818,
            28.26386056,
            28.26386056,
            28.26386056,
            28.23158201,
            25.26139038,
            25.26139038,
            22.8891583,
            22.8891583,
            20.91158092,
        ]
        tput_dict = {}
        for i in range(len(tputs)):
            tput_dict[30 - i] = tputs[i]

        if num_active_instances >= 30:
            self.step_delta = tput2iter_time_ms(tput_dict[30], batch_size)
        elif num_active_instances == 29:
            self.step_delta = tput2iter_time_ms(tput_dict[29], batch_size)
        elif num_active_instances == 28:
            self.step_delta = tput2iter_time_ms(tput_dict[28], batch_size)
        elif num_active_instances == 27:
            self.step_delta = tput2iter_time_ms(tput_dict[27], batch_size)
        elif num_active_instances == 26:
            self.step_delta = tput2iter_time_ms(tput_dict[26], batch_size)
        elif num_active_instances == 25:
            self.step_delta = tput2iter_time_ms(tput_dict[25], batch_size)
        elif num_active_instances == 24:
            self.step_delta = tput2iter_time_ms(tput_dict[24], batch_size)
        elif num_active_instances == 23:
            self.step_delta = tput2iter_time_ms(tput_dict[23], batch_size)
        elif num_active_instances == 22:
            self.step_delta = tput2iter_time_ms(tput_dict[22], batch_size)
        elif num_active_instances == 21:
            self.step_delta = tput2iter_time_ms(tput_dict[21], batch_size)
        elif num_active_instances == 20:
            self.step_delta = tput2iter_time_ms(tput_dict[20], batch_size)
        elif num_active_instances == 19:
            self.step_delta = tput2iter_time_ms(tput_dict[19], batch_size)
        elif num_active_instances == 18:
            self.step_delta = tput2iter_time_ms(tput_dict[18], batch_size)
        elif num_active_instances == 17:
            self.step_delta = tput2iter_time_ms(tput_dict[17], batch_size)
        elif num_active_instances == 16:
            self.step_delta = tput2iter_time_ms(tput_dict[16], batch_size)
        elif num_active_instances <= 15:
            self.step_delta = tput2iter_time_ms(tput_dict[15], batch_size)
        else:
            raise NotImplementedError

    def gpt_3_2_7b_simulate_step_delta(self):
        # set up self.step_delta base on number of active instances
        batch_size = self.samples_per_step # should match the batch size above
        num_active_instances = 0
        for name, instance in self.spot_instances.items():
            if instance.is_running() or instance.is_ready():
                num_active_instances += 1
        
        idle_gpu_overhead = {
                    30: 2,
                    29: 2,
                    28: 0,
                    27: 3,
                    26: 2,
                    25: 1,
                    24: 0,
                    23: 3,
                    22: 2,
                    21: 1,
                    20: 0,
                    19: 3,
                    18: 2,
                    17: 1,
                    16: 0,
                    15: 3,
                }
        if num_active_instances in idle_gpu_overhead.keys():
            self.curr_gpu_overhead = idle_gpu_overhead[num_active_instances]
        else:
            self.curr_gpu_overhead = 0.
        
        # milliseconds
        tputs = [
            9.767532721,
            9.424740405,
            9.424740405,
            9.099911677,
            9.099911677,
            9.099911677,
            9.099911677,
            7.469736582,
            7.469736582,
            7.469736582,
            7.469736582,
            6.424614304,
            6.424614304,
            6.424614304,
            6.424614304,
            4.863785398,
        ]
        tput_dict = {}
        for i in range(len(tputs)):
            tput_dict[30 - i] = tputs[i]

        if num_active_instances >= 30:
            self.step_delta = tput2iter_time_ms(tput_dict[30], batch_size)
        elif num_active_instances == 29:
            self.step_delta = tput2iter_time_ms(tput_dict[29], batch_size)
        elif num_active_instances == 28:
            self.step_delta = tput2iter_time_ms(tput_dict[28], batch_size)
        elif num_active_instances == 27:
            self.step_delta = tput2iter_time_ms(tput_dict[27], batch_size)
        elif num_active_instances == 26:
            self.step_delta = tput2iter_time_ms(tput_dict[26], batch_size)
        elif num_active_instances == 25:
            self.step_delta = tput2iter_time_ms(tput_dict[25], batch_size)
        elif num_active_instances == 24:
            self.step_delta = tput2iter_time_ms(tput_dict[24], batch_size)
        elif num_active_instances == 23:
            self.step_delta = tput2iter_time_ms(tput_dict[23], batch_size)
        elif num_active_instances == 22:
            self.step_delta = tput2iter_time_ms(tput_dict[22], batch_size)
        elif num_active_instances == 21:
            self.step_delta = tput2iter_time_ms(tput_dict[21], batch_size)
        elif num_active_instances == 20:
            self.step_delta = tput2iter_time_ms(tput_dict[20], batch_size)
        elif num_active_instances == 19:
            self.step_delta = tput2iter_time_ms(tput_dict[19], batch_size)
        elif num_active_instances == 18:
            self.step_delta = tput2iter_time_ms(tput_dict[18], batch_size)
        elif num_active_instances == 17:
            self.step_delta = tput2iter_time_ms(tput_dict[17], batch_size)
        elif num_active_instances == 16:
            self.step_delta = tput2iter_time_ms(tput_dict[16], batch_size)
        elif num_active_instances <= 15:
            self.step_delta = tput2iter_time_ms(tput_dict[15], batch_size)
        else:
            raise NotImplementedError

    def gpt_3_6_7b_simulate_step_delta(self):
        # set up self.step_delta base on number of active instances
        batch_size = self.samples_per_step # should match the batch size above
        num_active_instances = 0
        for name, instance in self.spot_instances.items():
            if instance.is_running() or instance.is_ready():
                num_active_instances += 1
        
        idle_gpu_overhead = {
                    30: 2,
                    29: 2,
                    28: 0,
                    27: 3,
                    26: 2,
                    25: 1,
                    24: 0,
                    23: 3,
                    22: 2,
                    21: 1,
                    20: 0,
                    19: 3,
                    18: 2,
                    17: 1,
                    16: 0,
                    15: 3,
                }
        if num_active_instances in idle_gpu_overhead.keys():
            self.curr_gpu_overhead = idle_gpu_overhead[num_active_instances]
        else:
            self.curr_gpu_overhead = 0.
        
        # milliseconds
        tputs = [
            5.240635953,
            5.240635953,
            5.240635953,
            5.096885803,
            5.096885803,
            5.096885803,
            5.096885803,
            4.503390788,
            4.503390788,
            4.503390788,
            4.503390788,
            3.953767179,
            3.953767179,
            3.953767179,
            3.953767179,
            3.016421755,
        ]
        tput_dict = {}
        for i in range(len(tputs)):
            tput_dict[30 - i] = tputs[i]

        if num_active_instances >= 30:
            self.step_delta = tput2iter_time_ms(tput_dict[30], batch_size)
        elif num_active_instances == 29:
            self.step_delta = tput2iter_time_ms(tput_dict[29], batch_size)
        elif num_active_instances == 28:
            self.step_delta = tput2iter_time_ms(tput_dict[28], batch_size)
        elif num_active_instances == 27:
            self.step_delta = tput2iter_time_ms(tput_dict[27], batch_size)
        elif num_active_instances == 26:
            self.step_delta = tput2iter_time_ms(tput_dict[26], batch_size)
        elif num_active_instances == 25:
            self.step_delta = tput2iter_time_ms(tput_dict[25], batch_size)
        elif num_active_instances == 24:
            self.step_delta = tput2iter_time_ms(tput_dict[24], batch_size)
        elif num_active_instances == 23:
            self.step_delta = tput2iter_time_ms(tput_dict[23], batch_size)
        elif num_active_instances == 22:
            self.step_delta = tput2iter_time_ms(tput_dict[22], batch_size)
        elif num_active_instances == 21:
            self.step_delta = tput2iter_time_ms(tput_dict[21], batch_size)
        elif num_active_instances == 20:
            self.step_delta = tput2iter_time_ms(tput_dict[20], batch_size)
        elif num_active_instances == 19:
            self.step_delta = tput2iter_time_ms(tput_dict[19], batch_size)
        elif num_active_instances == 18:
            self.step_delta = tput2iter_time_ms(tput_dict[18], batch_size)
        elif num_active_instances == 17:
            self.step_delta = tput2iter_time_ms(tput_dict[17], batch_size)
        elif num_active_instances == 16:
            self.step_delta = tput2iter_time_ms(tput_dict[16], batch_size)
        elif num_active_instances <= 15:
            self.step_delta = tput2iter_time_ms(tput_dict[15], batch_size)
        else:
            raise NotImplementedError

    def gpt_2_simulate_step_delta(self):
        # print(self.num_pipelines, self.num_stages)
        # sleep(10)
        if self.num_pipelines == 4 and self.num_stages == 12:
            self.step_delta = 1_008  # milliseconds
        elif self.num_pipelines == 3 and self.num_stages == 12:
            self.step_delta = 1_046  # milliseconds
        elif self.num_pipelines == 2 and self.num_stages == 12:
            self.step_delta = 1_085  # milliseconds
        elif self.num_pipelines == 1 and self.num_stages == 12:
            self.step_delta = 1_200  # milliseconds
        else:
            raise NotImplementedError

    def bert_simulate_step_delta(self):
        if self.num_pipelines == 4 and self.num_stages == 8:
            self.step_delta = 1_770  # milliseconds
        elif self.num_pipelines == 3 and self.num_stages == 8:
            self.step_delta = 2_300  # milliseconds
        elif self.num_pipelines == 2 and self.num_stages == 8:
            self.step_delta = 3_100  # milliseconds
        elif self.num_pipelines == 1 and self.num_stages == 8:
            self.step_delta = 5_700  # milliseconds
        elif self.num_pipelines == 4 and self.num_stages == 12:
            self.step_delta = 1_450  # milliseconds
        elif self.num_pipelines == 3 and self.num_stages == 12:
            self.step_delta = 1_805  # milliseconds
        elif self.num_pipelines == 2 and self.num_stages == 12:
            self.step_delta = 2_390  # milliseconds
        elif self.num_pipelines == 1 and self.num_stages == 12:
            self.step_delta = 4_310  # milliseconds
        elif self.num_pipelines == 4 and self.num_stages == 15:
            self.step_delta = 3_700  # milliseconds
        elif self.num_pipelines == 3 and self.num_stages == 15:
            self.step_delta = 4_780  # milliseconds
        elif self.num_pipelines == 2 and self.num_stages == 15:
            self.step_delta = 6_600  # milliseconds
        elif self.num_pipelines == 1 and self.num_stages == 15:
            self.step_delta = 12_750  # milliseconds
        else:
            raise NotImplementedError

    def resnet_simulate_step_delta(self):
        if self.num_pipelines == 4 and self.num_stages == 8:
            self.step_delta = 1_600  # milliseconds
        elif self.num_pipelines == 3 and self.num_stages == 8:
            self.step_delta = 2_400  # milliseconds
        elif self.num_pipelines == 2 and self.num_stages == 8:
            self.step_delta = 3_200  # milliseconds
        elif self.num_pipelines == 1 and self.num_stages == 8:
            self.step_delta = 6_100  # milliseconds
        else:
            raise NotImplementedError

    def gnmt_simulate_step_delta(self):
        if self.num_pipelines == 4 and self.num_stages == 6:
            self.step_delta = 950  # milliseconds
        elif self.num_pipelines == 3 and self.num_stages == 6:
            self.step_delta = 1_185  # milliseconds
        elif self.num_pipelines == 2 and self.num_stages == 6:
            self.step_delta = 1_660  # milliseconds
        elif self.num_pipelines == 1 and self.num_stages == 6:
            self.step_delta = 3_100  # milliseconds
        else:
            raise NotImplementedError

    def vgg_simulate_step_delta(self):
        if self.num_pipelines == 4 and self.num_stages == 6:
            self.step_delta = 2_750  # milliseconds
        elif self.num_pipelines == 3 and self.num_stages == 6:
            self.step_delta = 3_440  # milliseconds
        elif self.num_pipelines == 2 and self.num_stages == 6:
            self.step_delta = 4_900  # milliseconds
        elif self.num_pipelines == 1 and self.num_stages == 6:
            self.step_delta = 9_300  # milliseconds
        else:
            raise NotImplementedError

    def alexnet_simulate_step_delta(self):
        if self.num_pipelines == 4 and self.num_stages == 6:
            self.step_delta = 1_550  # milliseconds
        elif self.num_pipelines == 3 and self.num_stages == 6:
            self.step_delta = 2_000  # milliseconds
        elif self.num_pipelines == 2 and self.num_stages == 6:
            self.step_delta = 3_100  # milliseconds
        elif self.num_pipelines == 1 and self.num_stages == 6:
            self.step_delta = 6_180  # milliseconds
        else:
            raise NotImplementedError

    def info(self, delta, message):
        logger.info(f"[{delta/1000.0:.3f}] {message}")

    def get_spot_instance_next_name(self):
        name = self.spot_instance_name_format.format(id=self.spot_instance_next_id)
        self.spot_instance_next_id += 1
        return name

    def generate_spot_instance_probability_delta(
        self, current_time, current_delta, probability
    ):
        hour = current_time.hour
        p = probability[hour]
        if p > self.r.random():
            local_delta = datetime.timedelta(seconds=self.r.randrange(0, 3600))
            delta = current_delta + local_delta
            return delta
        return None

    def generate_spot_instance_addition_delta(self, current_time, current_delta):
        return self.generate_spot_instance_probability_delta(
            current_time, current_delta, self.spot_instance_addition_probability
        )

    def generate_spot_instance_removal_delta(self, current_time, current_delta):
        return self.generate_spot_instance_probability_delta(
            current_time, current_delta, self.spot_instance_removal_probability
        )

    def create_event(self, delta, kind, data):
        if isinstance(delta, datetime.timedelta):
            event = Event(delta // self.millisecond, kind, data)
        else:
            assert type(delta) == int
            event = Event(delta, kind, data)
        heapq.heappush(self.events, event)
        return event

    def create_spot_instance_generate_event(self, delta):
        return self.create_event(delta, EventKind.SPOT_INSTANCE_GENERATE, {})

    def create_spot_instance_add_event(self, delta, name=None):
        if name is None:
            name = self.get_spot_instance_next_name()
        return self.create_event(
            delta,
            EventKind.SPOT_INSTANCE_ADD,
            {
                "name": name,
            },
        )

    def create_spot_instance_remove_event(self, delta, name):
        return self.create_event(
            delta,
            EventKind.SPOT_INSTANCE_REMOVE,
            {
                "name": name,
            },
        )

    def create_spot_instance_ready_event(self, delta, name):
        return self.create_event(
            delta,
            EventKind.SPOT_INSTANCE_READY,
            {
                "name": name,
            },
        )

    def create_global_rendezvous_timeout_event(self, delta):
        if self.no_restart_overhead:
            self.local_rendezvous_timeout_delta = 0
            self.global_rendezvous_timeout_delta = 0
            self.checkpointing_overhead = 0

        self.restart_time += (
            self.local_rendezvous_timeout_delta - self.checkpointing_overhead
        )
        return self.create_event(
            # delta + self.global_rendezvous_timeout_delta + self.local_rendezvous_timeout_delta,
            delta + self.local_rendezvous_timeout_delta - self.checkpointing_overhead,
            EventKind.GLOBAL_RENDEZVOUS_TIMEOUT,
            {},
        )

    def create_checkpointing_event(self, delta):
        if self.no_ckpt_overhead:
            # immediately complete checkpointing
            return  # self.create_event(delta, EventKind.CHECKPOINT_COMPLETE,{})

        self.checkpoint_time += self.checkpointing_overhead
        return self.create_event(
            delta + self.checkpointing_overhead, EventKind.CHECKPOINT_COMPLETE, {}
        )

    def create_local_rendezvous_timeout_event(self, delta):
        if self.no_restart_overhead:
            self.local_rendezvous_timeout_delta = 0
            self.global_rendezvous_timeout_delta = 0
            self.checkpointing_overhead = 0

        return self.create_event(
            delta + self.local_rendezvous_timeout_delta,
            EventKind.LOCAL_RENDEZVOUS_TIMEOUT,
            {},
        )

    def create_training_step_complete_event(self, delta, rendezvous_version):
        self.effective_time += self.step_delta
        return self.create_event(
            delta + self.step_delta,
            EventKind.TRAINING_STEP_COMPLETE,
            {"rendezvous_version": rendezvous_version},
        )

    def create_training_step_complete_event_absolute(self, delta, rendezvous_version):
        return self.create_event(
            delta,
            EventKind.TRAINING_STEP_COMPLETE,
            {"rendezvous_version": rendezvous_version},
        )

    def generate_spot_instance_initial_events(self, start):
        # Generate the initial instances
        spot_instance_initial_probability = self.spot_instance_addition_probability[
            start.hour
        ]
        delta = 0
        for i in range(self.spot_instance_desired_capacity):
            if spot_instance_initial_probability > self.r.random():
                event = self.create_spot_instance_add_event(delta)
        self.create_spot_instance_generate_event(delta)

    # def generate_spot_instance_events(self, start, duration): # TODO
    def generate_spot_instance_events(self, start, delta):
        current_delta = delta * self.millisecond
        self.create_spot_instance_generate_event(current_delta + self.hour)

        current_instances = {}
        for name, instance in self.spot_instances.items():
            current_instances[name] = current_delta

        # The remaining code generates add and remove events for the next hour
        current_time = start + current_delta

        # Run removal for currently running nodes
        removed_instances = []
        for name in current_instances.keys():
            delta = self.generate_spot_instance_removal_delta(
                current_time, current_delta
            )
            if delta is None:
                continue
            removed_instances.append((delta, name))
        heapq.heapify(removed_instances)

        # Run additions for the maximum number of instances you ever need
        # this hour
        possible_added_deltas = []
        requested_capacity = (
            self.spot_instance_desired_capacity
            - len(current_instances)
            + len(removed_instances)
        )
        for _ in range(requested_capacity):
            delta = self.generate_spot_instance_addition_delta(
                current_time, current_delta
            )
            if delta is None:
                continue
            possible_added_deltas.append(delta)
        heapq.heapify(possible_added_deltas)

        # Check if we have to throw out any additions because we're already
        # at capacity
        while len(removed_instances) > 0 or len(possible_added_deltas) > 0:
            # The next event is a removal, so just do it
            if len(removed_instances) > 0 and (
                len(possible_added_deltas) == 0
                or removed_instances[0][0] < possible_added_deltas[0]
            ):
                delta, name = heapq.heappop(removed_instances)

                event = self.create_spot_instance_remove_event(delta, name)

                del current_instances[name]
            # The next event is an addition, only do it if we're under
            # desired capacity
            else:
                delta = heapq.heappop(possible_added_deltas)

                # Skip this addition
                if len(current_instances) == self.spot_instance_desired_capacity:
                    continue

                event = self.create_spot_instance_add_event(delta)

                name = event.data["name"]
                current_instances[name] = delta

                # Check if we also attempt to remove this new instance in
                # this hour
                delta = self.generate_spot_instance_removal_delta(
                    current_time, current_delta
                )
                if delta is None or delta < current_instances[name]:
                    continue
                heapq.heappush(removed_instances, (delta, name))

    def simulate_spot_instance_add(self, delta, data):
        name = data["name"]
        self.spot_instances[name] = SpotInstance(name, delta)
        self.create_spot_instance_ready_event(
            delta + self.spot_instance_creation_time,
            name,
        )

    def simulate_fatal_failure(self, delta, name):
        self.info(delta, f"{name} caused a fatal failure, starting global rendezvous")
        self.num_fatal_failures += 1
        self.simulate_rendezvous_start(delta, True)

    def simulate_spot_instance_remove(self, delta, data):
        name = data["name"]
        instance = self.spot_instances[name]

        self.num_spot_instance_removals += 1
        self.spot_instance_lifetimes.append(delta - instance.start)
        self.spot_instance_removal_times.append(delta)
        del self.spot_instances[name]

        # fall back to previous checkpoint
        self.node_will_be_removed = True

        if not self.no_fallback:
            if self.prev_checkpoint_delta == -1:
                # print("stuck...")
                # stuck at the begining of the training, no progress
                self.performance_ys[:] = [0.0] * len(self.performance_ys[:])
                self.fallback_loss_time += delta
            else:
                for i, (t, p) in enumerate(zip(self.performance_xs, self.performance_ys)):
                    wasted_p_idx = i
                    if t == self.prev_checkpoint_delta:
                        # print("wasted:", wasted_p_idx)
                        break
                self.performance_ys[wasted_p_idx + 1 :] = [0.0] * len(
                    self.performance_ys[wasted_p_idx + 1 :]
                )
                # self.fallback_loss_time += delta - self.prev_checkpoint_delta
                self.fallback_loss_time += (self.step_delta * len(self.performance_ys[wasted_p_idx+1:]))

        self.simulate_fatal_failure(delta, name)
        return
        # if instance.is_running():
        #     # Full restart
        #     self.simulate_fatal_failure(delta, name)
        #     return

    def simulate_rendezvous_start(self, delta, is_global):
        self.rendezvous_start_counter += 1
        self.status = SystemStatus.RENDEZVOUS
        self.simulate_rendezvous_restart(delta)
        if is_global:
            self.create_global_rendezvous_timeout_event(delta)
        else:
            self.create_local_rendezvous_timeout_event(delta)

    def simulate_rendezvous_restart(self, delta):
        assert self.status == SystemStatus.RENDEZVOUS
        self.rendezvous = []
        for (
            name,
            instance,
        ) in self.spot_instances.items():
            if instance.is_ready() or instance.is_running():
                self.rendezvous.append(name)

    def simulate_rendezvous_timeout(self, delta, is_global):
        for i, name in enumerate(self.rendezvous):
            if name not in self.spot_instances:
                self.info(delta, f"{name} terminated during redezvous, restarting")
                self.simulate_rendezvous_restart(delta)
                if is_global:
                    self.create_global_rendezvous_timeout_event(delta)
                else:
                    self.create_local_rendezvous_timeout_event(delta)
                return
            instance = self.spot_instances[name]
            instance.global_id = i
        self.simulate_assign_coordinates()
        self.fallback_event = None
        self.fallback_handled = False
        self.rendezvous_version += 1
        self.info(
            delta,
            f"{len(self.rendezvous)} nodes completed rendezvous version "
            f"{self.rendezvous_version}, "
            f"{self.num_pipelines}x{self.num_stages} configuration",
        )
        self.rendezvous = []
        if self.num_pipelines != 0:
            self.status = SystemStatus.RUNNING
            self.simulate_step_delta()
            self.create_training_step_complete_event(delta, self.rendezvous_version)
            if self.start_delta is None:
                self.start_delta = delta
                self.previous_step_complete_delta = self.start_delta
        else:
            self.status = SystemStatus.STOPPED

    def simulate_spot_instance_ready(self, delta, data):
        name = data["name"]

        # This node has already been removed
        if name not in self.spot_instances:
            return

        instance = self.spot_instances[name]
        instance.set_ready()

        if self.status == SystemStatus.STOPPED:
            self.info(delta, f"{name} starting global rendezvous")
            self.simulate_rendezvous_start(delta, True)
        elif self.status == SystemStatus.RENDEZVOUS:
            self.rendezvous.append(name)
        elif self.status == SystemStatus.RUNNING:
            self.num_workers_waiting += 1

    def simulate_global_rendezvous_timeout(self, delta, data):
        self.simulate_rendezvous_timeout(delta, True)

    def simulate_local_rendezvous_timeout(self, delta, data):
        self.simulate_rendezvous_timeout(delta, False)

    def simulate_assign_coordinates(self):
        if len(self.rendezvous) < self.num_stages_target:
            num_pipelines = 0
            num_stages = 0
        else:
            num_stages = self.num_stages_target
            num_pipelines = len(self.rendezvous) // num_stages
        num_workers_waiting = 0

        previous_num_pipelines = self.num_pipelines
        previous_num_stages = self.num_stages

        required_coordinates = []
        for i in range(num_pipelines):
            for j in range(num_stages):
                required_coordinates.append((i, j))

        for name in self.rendezvous:
            instance = self.spot_instances[name]
            assert instance.is_ready() or instance.is_running()
            instance.previous_coordinates = instance.active_coordinates
            try:
                coordinates = required_coordinates.pop(0)
                instance.active_coordinates = [coordinates]
                instance.set_running()
            except IndexError:
                instance.active_coordinates = []
                instance.set_ready()
                num_workers_waiting += 1

        self.num_pipelines = num_pipelines
        self.num_stages = num_stages
        self.num_workers_waiting = num_workers_waiting

    def get_num_workers_overloaded(self):
        num_workers_overloaded = 0
        for name, instance in self.spot_instances.items():
            if not instance.is_running():
                continue
            elif len(instance.active_coordinates) > 1:
                assert len(instance.active_coordinates) == 2
                num_workers_overloaded += 1
        return num_workers_overloaded

    def simulate_should_reconfigure(self):
        return False
        num_workers_overloaded = self.get_num_workers_overloaded()
        num_workers_waiting = self.num_workers_waiting

        # If we can easily re-balance, do it
        if num_workers_overloaded > 0 and num_workers_waiting >= num_workers_overloaded:
            return True

        num_original_workers = self.num_pipelines * self.num_stages
        num_active_workers = num_original_workers - num_workers_overloaded

        # If we're above a 5% chance of failure, re-configure/re-balance
        # I cannot remove an overloaded node, or the node that now has no
        # redundancy
        if (
            num_workers_overloaded > 0
            and (2 * num_workers_overloaded / num_active_workers) > 0.05
        ):
            return True

        # If we can add another pipeline, do it
        potential_num_pipelines = (
            num_active_workers + num_workers_waiting
        ) // self.num_stages_target
        if potential_num_pipelines > self.num_pipelines:
            return True

        return False

    def checkpointing_will_fail(self):
        # loop the current event list and if there is a checkpoint event
        remove_events_interput = False
        for e in self.events:
            if e.kind == EventKind.SPOT_INSTANCE_REMOVE:
                remove_events_interput = True

            if e.kind == EventKind.CHECKPOINT_COMPLETE:
                if not remove_events_interput:
                    return False
                else:
                    return True
        return False

    def simulate_training_step_complete(self, delta, data):
        rendezvous_version = data["rendezvous_version"]

        if rendezvous_version != self.rendezvous_version:
            return

        # Handle fallback events
        # if self.fallback_event is not None:
        #     event_num_steps_complete, event_delta = self.fallback_event
        #     if not self.fallback_handled and event_num_steps_complete == self.num_steps_complete:
        #         # The duration we need to add to handle the fallback
        #         d = int((delta - event_delta) * (self.fallback_slowdown - 1.0))
        #         self.create_training_step_complete_event_absolute(
        #             d + delta,
        #             rendezvous_version
        #         )
        #         self.fallback_handled = True
        #         return

        self.num_steps_complete += 1

        # Calculate performance
        step_duration = delta - self.previous_step_complete_delta  # milliseconds
        # print('Step duration:', step_duration)

        step_duration_seconds = step_duration / self.milliseconds_per_second
        step_duration_hours = step_duration / self.milliseconds_per_hour
        # print('Step duration (s):', step_duration_seconds)
        samples_per_second = self.samples_per_step / step_duration_seconds

        previous_delta_hours = (
            self.previous_step_complete_delta / self.milliseconds_per_hour
        )
        delta_hours = delta / self.milliseconds_per_hour

        self.performance_xs.append(previous_delta_hours)
        self.performance_ys.append(samples_per_second)
        self.performance_xs.append(delta_hours)
        self.performance_ys.append(samples_per_second)

        # GPU overhead
        num_active_instances = 0
        for name, instance in self.spot_instances.items():
            if instance.is_running() or instance.is_ready():
                num_active_instances += 1
        self.sum_gpu_overhead += (step_duration * self.curr_gpu_overhead) / num_active_instances
        # wasted_tput = [self.curr_gpu_overhead*samples_per_second, ....]
        # num_active_instances_at = [num_active_instances, .....]
        # ==> sum(wasted_tput) / sum(num_active_instances_at)

        # self.checkpoint_per_iter = 10
        if self.num_steps_complete % self.checkpoint_per_iter == 0 or self.node_will_be_removed:
            self.node_will_be_removed = False
            if not self.no_ckpt_overhead:
                delta += self.checkpointing_overhead
                delta_hours = delta / self.milliseconds_per_hour

            self.create_checkpointing_event(delta)
            # check if checkpoint complete is after this remove event
            heapq.heapify(self.events)
            if self.checkpointing_will_fail() and not self.no_fallback:
                # self.fallback_loss_time += delta - self.prev_checkpoint_delta
                self.fallback_loss_time += ( self.checkpoint_per_iter * self.step_delta )
                # print("fall back!!")
                self.num_steps_complete -= self.checkpoint_per_iter
                if self.num_steps_complete < 0:
                    self.num_steps_complete = 0
                # pprint(self.events)
                self.performance_ys[-self.checkpoint_per_iter * 2 :] = [0.0] * (
                    self.checkpoint_per_iter * 2
                )
                # print(len(self.performance_ys))
                # print(self.performance_ys[-15:])
                # print()
                # sleep(10)
            else:
                self.prev_checkpoint_delta = delta_hours

        current_cost_per_hour = self.cost_ys[-1]
        current_cost_delta = delta_hours
        total_cost = 0.0
        i = -1
        while True:
            x1 = self.cost_xs[i]
            y1 = self.cost_ys[i]
            try:
                x2 = self.cost_xs[i - 1]
                y2 = self.cost_ys[i - 1]
            except IndexError:
                total_cost += current_cost_per_hour * (
                    current_cost_delta - previous_delta_hours
                )
                break
            assert x1 == x2

            if x1 > previous_delta_hours:
                total_cost += current_cost_per_hour * (current_cost_delta - x1)
                current_cost_per_hour = y2
                current_cost_delta = x1
            else:
                total_cost += current_cost_per_hour * (
                    current_cost_delta - previous_delta_hours
                )
                break
            i -= 2

        average_cost_per_hour = total_cost / step_duration_hours

        self.value_xs.append(previous_delta_hours)
        self.value_ys.append(samples_per_second / average_cost_per_hour)
        self.value_xs.append(delta_hours)
        self.value_ys.append(samples_per_second / average_cost_per_hour)

        self.previous_step_complete_delta = delta

        if self.num_steps_complete % 100 == 0:
            self.info(delta, f"{self.num_steps_complete} steps complete")
        # if self.simulate_should_reconfigure():
        #     self.should_reconfigure_counter +=1
        #     self.info(
        #         delta,
        #         f'reconfiguration after step {self.num_steps_complete}'
        #     )
        #     self.simulate_rendezvous_start(delta, False)
        # else:
        #     self.create_training_step_complete_event(
        #         delta,
        #         self.rendezvous_version
        #     )
        self.create_training_step_complete_event(delta, self.rendezvous_version)

    def calculate_average(self, xs, ys, duration):
        if duration == 0:
            return 0.0
        previous_x = None
        previous_y = None
        total = 0.0
        for x, y in zip(xs, ys):
            if previous_x is None:
                previous_x = x
                previous_y = y
            else:
                # assert previous_y == y
                total += (x - previous_x) * y
                previous_x = None
                previous_y = None
        return total / duration

    def calculate_average_old(self, xs, ys, duration):
        print("=== WARNING OLD")
        previous_x = 0.0
        previous_y = 0.0
        total = 0.0
        for x, y in zip(xs, ys):
            total += (x - previous_x) * previous_y
            previous_x = x
            previous_y = y
        total += (duration - previous_x) * previous_y
        return total / duration

    def simulate(self, duration=None):
        start = datetime.datetime.now(datetime.timezone.utc)
        start = start.replace(minute=0, second=0, microsecond=0)
        if self.start_hour is not None:
            start = start.replace(hour=self.start_hour)

        logger.info(f"Starting at {start}")

        if self.spot_instance_trace is None:
            logger.info(f"Generating spot instance events...")
            self.generate_spot_instance_initial_events(start)
            # print(self.events)
        else:
            reader = csv.reader(self.spot_instance_trace)
            for row in reader:
                delta, event, name = row
                delta = int(delta)
                if event == "add":
                    self.create_spot_instance_add_event(delta, name)
                elif event == "remove":
                    self.create_spot_instance_remove_event(delta, name)
                else:
                    raise NotImplementedError

        instances_xs = []
        instances_ys = []
        self.performance_xs = []
        self.performance_ys = []
        self.cost_xs = []
        self.cost_ys = []
        self.value_xs = []
        self.value_ys = []

        heapq.heapify(self.events)
        self.last_event = self.events[-1]
        # print(self.last_event)
        # sleep(10)

        while len(self.events) > 0:
            event = heapq.heappop(self.events)

            kind = event.kind
            delta = event.delta
            data = event.data

            # time based hard stop
            if delta >= self.last_event.delta and self.time_hard_stop:
                break

            if duration is not None and delta > duration:
                delta = duration
                break

            if kind == EventKind.SPOT_INSTANCE_ADD:
                self.simulate_spot_instance_add(delta, data)
            elif kind == EventKind.SPOT_INSTANCE_REMOVE:
                self.simulate_spot_instance_remove(delta, data)
            elif kind == EventKind.SPOT_INSTANCE_GENERATE:
                self.generate_spot_instance_events(start, delta)
            elif kind == EventKind.SPOT_INSTANCE_READY:
                self.simulate_spot_instance_ready(delta, data)
            elif kind == EventKind.GLOBAL_RENDEZVOUS_TIMEOUT:
                self.simulate_global_rendezvous_timeout(delta, data)
            elif kind == EventKind.LOCAL_RENDEZVOUS_TIMEOUT:
                self.simulate_local_rendezvous_timeout(delta, data)
            elif kind == EventKind.TRAINING_STEP_COMPLETE:
                self.simulate_training_step_complete(delta, data)
            elif kind == EventKind.CHECKPOINT_COMPLETE:
                pass
            else:
                raise ValueError(f"Unknown kind: {kind}")

            # We're done our training
            if duration is None and self.num_steps_complete == self.steps_per_run:
                break

            # We still need to process more events for this delta
            next_event = self.events[0] if len(self.events) > 0 else None
            next_delta = next_event.delta if next_event else None
            if delta == next_delta:
                continue

            # Initialize the number of instances and cost
            if len(instances_xs) == 0 and delta > 0:
                num_instances = len(self.spot_instances)
                instances_xs.append(0)
                instances_ys.append(num_instances)
                self.cost_xs.append(0)
                self.cost_ys.append(num_instances * self.spot_instance_cost_per_hour)
            elif len(instances_xs) > 0:
                previous_num_instances = instances_ys[-1]
                num_instances = len(self.spot_instances)
                if previous_num_instances != num_instances:
                    delta_hours = delta / self.milliseconds_per_hour
                    instances_xs.append(delta_hours)
                    instances_ys.append(previous_num_instances)
                    self.cost_xs.append(delta_hours)
                    self.cost_ys.append(
                        previous_num_instances * self.spot_instance_cost_per_hour
                    )
                    instances_xs.append(delta_hours)
                    instances_ys.append(num_instances)
                    self.cost_xs.append(delta_hours)
                    self.cost_ys.append(
                        num_instances * self.spot_instance_cost_per_hour
                    )

        duration_hours_whole = math.ceil(delta / self.milliseconds_per_hour)

        if len(self.performance_xs) == 0:
            duration_hours = 0.0
        else:
            duration_hours = self.performance_xs[-1]
        num_instances = len(self.spot_instances)
        instances_xs.append(duration_hours)
        instances_ys.append(num_instances)
        self.cost_xs.append(duration_hours)
        self.cost_ys.append(num_instances * self.spot_instance_cost_per_hour)

        # Complete the remaining
        for name, instance in self.spot_instances.items():
            self.spot_instance_lifetimes.append(delta - instance.start)

        spot_instance_between_removal_times = []
        previous_removal_time = (
            self.spot_instance_removal_times[0]
            if len(self.spot_instance_removal_times) > 0
            else None
        )
        for removal_time in self.spot_instance_removal_times[1:]:
            spot_instance_between_removal_times.append(
                removal_time - previous_removal_time
            )
            previous_removal_time = removal_time

        performance_value_duration_hours = duration_hours - (
            self.start_delta / self.milliseconds_per_hour
        )

        result = Result(
            removal_probability=self.removal_probability,
            preemption_mean=statistics.mean(spot_instance_between_removal_times)
            / self.milliseconds_per_hour
            if len(spot_instance_between_removal_times) > 0
            else 0,
            preemption_median=statistics.mean(spot_instance_between_removal_times)
            / self.milliseconds_per_hour
            if len(spot_instance_between_removal_times) > 0
            else 0,
            preemption_stdev=statistics.stdev(spot_instance_between_removal_times)
            / self.milliseconds_per_hour
            if len(spot_instance_between_removal_times) > 1
            else 0,
            lifetime_mean=statistics.mean(self.spot_instance_lifetimes)
            / self.milliseconds_per_hour,
            lifetime_median=statistics.median(self.spot_instance_lifetimes)
            / self.milliseconds_per_hour,
            lifetime_stdev=statistics.stdev(self.spot_instance_lifetimes)
            / self.milliseconds_per_hour,
            num_preemptions=self.num_spot_instance_removals,
            num_fatal_failures=self.num_fatal_failures,
            num_steps_complete=self.num_steps_complete,
            average_instances=self.calculate_average(
                instances_xs, instances_ys, duration_hours
            ),
            average_performance=self.calculate_average(
                self.performance_xs,
                self.performance_ys,
                performance_value_duration_hours,
            ),
            average_cost=self.calculate_average(
                self.cost_xs, self.cost_ys, duration_hours
            ),
            average_value=self.calculate_average(
                self.value_xs, self.value_ys, performance_value_duration_hours
            ),
        )

        detailed_result = {}
        detailed_result["duration_hours_whole"] = duration_hours_whole
        detailed_result["instances"] = [
            instances_xs,
            instances_ys,
            result.average_instances,
        ]
        detailed_result["performance"] = [
            self.performance_xs,
            self.performance_ys,
            result.average_performance,
        ]
        detailed_result["cost"] = [self.cost_xs, self.cost_ys, result.average_cost]
        detailed_result["value"] = [self.value_xs, self.value_ys, result.average_value]

        print("varuna")
        # print(f"Ending delta: {delta}")
        total = self.effective_time + self.checkpoint_time + self.restart_time + \
                self.sum_gpu_overhead + self.fallback_loss_time
        print(
            f"Sum of breakdowns: {total}"
        )
        print("Effective Time:{}, {}".format(self.effective_time, self.effective_time/total))
        print("checkpoint_time:{}, {}".format(self.checkpoint_time, self.checkpoint_time/total))
        print("restart_time:{}, {}".format(self.restart_time, self.restart_time/total))
        print("Fallbacks:{}, {}".format(self.fallback_loss_time, self.fallback_loss_time/total))
        print("Sum idle_gpu_overhead:{}, {}".format(self.sum_gpu_overhead, self.sum_gpu_overhead/total))
        # print("step:", self.num_steps_complete)
        print()

        if self.generate_graphs:
            exit()
            from main import graph

            # pdf_suffix = f'-seed-{self.seed}-start-hour-{self.start_hour}-generate-addition-probabilities-{self.generate_addition_probabilities}-removal-probability-{self.removal_probability}.pdf'
            pdf_suffix = f"-{self.model}.pdf"

            # Instances graph
            graph(
                "Time (hours)",
                instances_xs,
                duration_hours_whole,
                "# Instances",
                instances_ys,
                max(self.on_demand_num_instances, max(instances_ys)),
                result.average_instances,
                on_demand=self.on_demand_num_instances,
                out=f"instances{pdf_suffix}",
                show=True,
            )

            # Performance graph
            graph(
                "Time (hours)",
                self.performance_xs,
                duration_hours_whole,
                "Performance (samples per second)",
                self.performance_ys,
                max(self.on_demand_performance, max(self.performance_ys)),
                result.average_performance,
                on_demand=self.on_demand_performance,
                out=f"performance{pdf_suffix}",
                show=True,
            )

            print("Model:", self.model)
            print(
                "  Performance:",
                "D",
                self.on_demand_performance,
                "B",
                result.average_performance,
            )

            # Cost graph
            graph(
                "Time (hours)",
                self.cost_xs,
                duration_hours_whole,
                "Cost ($ per hour)",
                self.cost_ys,
                max(self.on_demand_cost, max(self.cost_ys)),
                result.average_cost,
                on_demand=self.on_demand_cost,
                out=f"cost{pdf_suffix}",
                show=True,
            )

            print("  Cost:", "D", self.on_demand_cost, "B", result.average_cost)

            # Value graph
            graph(
                "Time (hours)",
                self.value_xs,
                duration_hours_whole,
                "Value (performance per cost)",
                self.value_ys,
                max(self.on_demand_value, max(self.value_ys)),
                result.average_value,
                on_demand=self.on_demand_value,
                out=f"value{pdf_suffix}",
                show=True,
            )

            print("  Value:", "D", self.on_demand_value, "B", result.average_value)

        # print('Preemptions')
        # print('  - Mean:', result.preemption_mean, 'hours')
        # print('  - Median:', result.preemption_median, 'hours')
        # print('  - Stdev:', result.preemption_stdev, 'hours')
        # print('Lifetimes')
        # print('  - Mean:', result.lifetime_mean, 'hours')
        # print('  - Median:', result.lifetime_median, 'hours')
        # print('  - Stdev:', result.lifetime_stdev, 'hours')
        # print('Number of preemptions:', result.num_preemptions)
        # print('Number of fatal failures:', result.num_fatal_failures)
        # print('Number of steps complete:', result.num_steps_complete)

        logger.info(f"Ending after {duration_hours} hours")

        # print(self.should_reconfigure_counter)
        # print(self.rendezvous_start_counter)
        # print(self.num_fatal_failures)
        if self.detail_res:
            return result, detailed_result
        else:
            return result
