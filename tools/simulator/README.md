# Reproducing results in the Paper

### Figure 9+13. Throughput with different frequency of failures. Note the different Y-axes scales for different models

```
bash eval_all.sh
```
This bash script will run simulations for 3 systems (Bamboo, Varuna and Oobleck) for 5 models (`BERT-large`, `GPT-2`, `GPT-3-medium`, `GPT-3-2_7b`, `GPT-3-6_7b`) on 5 traces with different failure rates (`6h`, `3h`, `1h`, `30m`, `10m`). The plots will be saved under the current directory as PDF files.

And it generates the figure 13: Comparison of Varuna, Varuna with no checkpointing
overhead, and Oobleck running the GPT-3 6.7b model. 


### Figure 10+14+15. Throughput changes in spot instances environment, EC2 P3 instances and GCP a2-highgpu-1g instances. 

```
python eval_gcp_ec2.py
```
This python script will run simulations for 3 systems (Bamboo, Varuna and Oobleck) for 5 models (`BERT-large`, `GPT-2`, `GPT-3-medium`, `GPT-3-2_7b`, `GPT-3-6_7b`) on 2 scaled real traces (GCP and EC2 p3). The plots will be saved under the current directory as PDF files.
