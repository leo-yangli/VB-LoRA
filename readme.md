# VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks

This repo contains the source code for [VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks](https://arxiv.org/pdf/2405.15179).

<img src="https://github.com/leo-yangli/VB-LoRA/blob/main/param_comp.png?raw=True" alt="Comparison of the PEFT methods" width="350"/>

***Comparison with other PEFT methods on RoBERTa-Large.** VB-LoRA achieves higher scores with significantly smaller number of stored parameters.*

![Overview](https://github.com/leo-yangli/VB-LoRA/blob/main/VB-LoRA.png?raw=True)
***Overview of VBLoRA**. Left: The model parameters can be represented as a composition of vectors from a vector bank, which is shared across sub-vectors, modules and layers. Right: Architecture of VB-LoRA. We use a top-k softmax function to select k vectors from the vector bank. The selected vectors are then pooled into a sub-vector, which is arranged at a desired position, forming the parameters of LoRA.*

## Steps to reproduce the results

## NLU
- Modified code for running experiments for Natural Language Understanding experiments.
- Adapted from [LoRA source code](https://github.com/microsoft/LoRA).
#### Create and activate conda env
```console
cd NLU
conda env create -f environment.yml
conda activate VB_LoRA_NLU
```
#### Install the pre-requisites
vb-lora:
```console
pip install -e ..
```
NLU:
```console
pip install -e .
```
#### Start the experiments
The scripts are located in the "NLU/scripts_vblora_all" and "NLU/scripts_vblora_qv" folders.

For example,
```console
cd NLU
./scripts_vblora_all/roberta_base_cola.sh
```


## Instruction Tuning

- The code for running Llama2 is adapted from [qlora source code](https://github.com/artidoro/qlora).
- We implement VB-LoRA within the framework of Huggingface [PEFT](https://github.com/huggingface/peft/). Our added code can be found under ```peft/src/peft/tuners/vblora```
- Fine-tuning the Llama2 model requires access to the model weights on HuggingFace. Ensure you have the access before running the code.

#### Create and activate conda env
```console
cd instruction_tuning
conda create -n instruction_tuning python==3.10
conda activate instruction_tuning
```
#### Install the pre-requisites
qlora:
```console
cd instruction_tuning
pip install -r requirements.txt
```
peft:
```console
cd peft
pip install -r requirements.txt
```
#### Start the experiments
The scripts are located in the "instruction_tuning/scripts" folder.

For example,
```console
cd instruction_tuning
./scripts/finetune_llama2_7b_vb.sh
```

For evaluation, please use [LLM Judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).


## Citation
If you found this code useful, please cite our paper.

```  
@misc{li2024vblora,
      title={VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks}, 
      author={Yang Li and Shaobo Han and Shihao Ji},
      year={2024},
      eprint={2405.15179},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```  
