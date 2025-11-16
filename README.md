# FYP

1. Evaluate Qwen3-VL-2B on AMBER, MMHalBench, ObjectHalBench, POPE. Try not to be too detailed, just get an understanding first, look at the scores, observe the mistakes using a targeted approach.

* Clone Qwen3-VL-2B to current env

  * Need to move the cache directory to the current directory.
    `export HF_HOME="/home/users/ntu/hilarius/scratch/fyp"` [DONE]
  * Make a new bash script to ensure the variable persists and inference is done on GPU.

    * Make a new job for this. [DONE]
    * Install flash attn 2.7.2 - cu11 torch 2.6, abi FALSE, cp9
      [https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu11torch2.6cxx11abiFALSE-cp39-cp39-linux_x86_64.whl](https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu11torch2.6cxx11abiFALSE-cp39-cp39-linux_x86_64.whl) [DONE]
  * Remember to use wandb for monitoring.

* Evaluate on these benchmarks, understanding the metrics and look at the cases where it failed.

2. Use OPA-DPO and try to analyze how it works and whether it fixed the problem.

Choices:

* Size of Qwen3 VL: 2B, 4B, 8B. Decide to choose 2B for simplicity.
* Choose thinking/instruct variant. Decide to choose instruct as we focus on hallucination, not reasoning.
* Tune the hyperparameters for answer generation.

Make sure to always update on GitHub every new progress is made.

