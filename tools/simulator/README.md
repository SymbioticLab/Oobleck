# Reproducing results in the Paper

### Figure 9+13: Throughput with different frequency of failures

```
bash eval_all.sh
```
This bash script will run simulations for 3 systems (Bamboo, Varuna and Oobleck) for 5 models (`BERT-large`, `GPT-2`, `GPT-3-medium`, `GPT-3-2_7b`, `GPT-3-6_7b`) on 5 traces with different failure rates (`6h`, `3h`, `1h`, `30m`, `10m`). The plots will be saved under the current directory as PDF files.

And it generates the figure 13: Comparison of Varuna, Varuna with no checkpointing
overhead, and Oobleck running the GPT-3 6.7b model. 


### Figure 10+14+15: Throughput changes in spot instances environment (EC2 P3 and GCP)

```
python eval_gcp_ec2.py
```
This python script will run simulations for 3 systems (Bamboo, Varuna and Oobleck) for 5 models (`BERT-large`, `GPT-2`, `GPT-3-medium`, `GPT-3-2_7b`, `GPT-3-6_7b`) on 2 scaled real traces (GCP and EC2 p3). The plots will be saved under the current directory as PDF files.


### Figure 1+12: Time occupation breakdown 

We monitor the time occupation of each component such as the time spent on checkpointing, reconfiguration in the system during the simulation. The effective time can be computed at the end of each simulation and will be printed out at the end of each simulation. 

```
python eval_tput_vs_freq.py -m BERT-large
```

```
python eval_tput_vs_freq.py -m GPT-3-6_7b
```

Example output (`BERT-large` simulation with failure happens every 10 minutes):

```
bamboo
Effective Time:1521824.4737999989, 0.1497772789022047
redundant_time:6199253.526199993, 0.6101277383591077
rebalance_time:912000, 0.08975862900780404
restart_time:30000, 0.00295258648051987
Fallback:1497505.0, 0.1473837672503636
Sum idle_gpu_overhead:0.0, 0.0

varuna
Effective Time:8513245, 0.5444607905179443
checkpoint_time:632000, 0.04041927838413446
restart_time:1048100, 0.06703076847216982
Fallbacks:5442758, 0.34808916262575146
Sum idle_gpu_overhead:0.0, 0.0

oobleck
Effective Time:9219611.0, 0.9551596805322267
rebalance_time:30000.0, 0.00310802596942179
restart_time:0.0, 0.0
Fallbacks:402818.0, 0.04173229349835156
Sum idle_gpu_overhead:0.0, 0.0
```