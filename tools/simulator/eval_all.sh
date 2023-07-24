#!/bin/sh

# BERT-large, GPT-2-ours, GPT-3-medium, GPT-3-2_7b, GPT-3-6_7b

python eval_tput_vs_freq.py -m BERT-large -s bert-large-eval.pdf

python eval_tput_vs_freq.py -m GPT-2-ours -s gpt2-eval.pdf

python eval_tput_vs_freq.py -m GPT-3-medium -s gpt3-medium-eval.pdf

python eval_tput_vs_freq.py -m GPT-3-2_7b -s gpt3-2_7b-eval.pdf

python eval_tput_vs_freq.py -m GPT-3-6_7b -s gpt3-6_7b-eval.pdf

# eval oobleck vs varuna with and without ckpt overhead
python eval_no_ckpt_overhead.py -m GPT-3-6_7b -s gpt3-6_7b-no_ckpt_overhead_eval.pdf
