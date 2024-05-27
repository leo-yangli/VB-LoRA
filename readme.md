## [NeurIPS 2024] VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks

This repo contains the source code for [VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks](https://arxiv.org/pdf/2405.15179).

VB-LoRA is now integrated in the [State-of-the-art Parameter-Efficient Fine-Tuning (PEFT)](https://github.com/huggingface/peft) library by Hugging Face. Please check the [doc](https://huggingface.co/docs/peft/main/en/package_reference/vblora), [code](https://github.com/huggingface/peft/tree/main/src/peft/tuners/vblora), and [examples](https://github.com/huggingface/peft/blob/main/examples/sequence_classification/VBLoRA.ipynb).

## Abstract
As the adoption of large language models increases and the need for per-user or per-task model customization grows, the parameter-efficient fine-tuning (PEFT) methods, such as low-rank adaptation (LoRA) and its variants, incur substantial storage and transmission costs. To further reduce stored parameters, we introduce a "divide-and-share" paradigm that breaks the barriers of low-rank decomposition across matrix dimensions, modules and layers by sharing parameters globally via a *vector bank*. As an instantiation of the paradigm to LoRA, our proposed VB-LoRA composites *all* the low-rank matrices of LoRA from a shared *vector bank* with a differentiable top-k admixture module. VB-LoRA achieves extreme parameter efficiency while maintaining comparable or better performance compared to state-of-the-art PEFT methods. Extensive experiments demonstrate the effectiveness of VB-LoRA on natural language understanding, natural language generation, and instruction tuning tasks. **When fine-tuning the Llama2-13B model, VB-LoRA only uses 0.4% of LoRA's stored parameters, yet achieves superior results.** 

<!---![Overview](https://github.com/leo-yangli/VB-LoRA/blob/main/VB-LoRA.png?raw=True)--->
<img src="https://github.com/leo-yangli/VB-LoRA/blob/main/VB-LoRA.png?raw=True" alt="Overview of VB-LoRA" width="100%"/>

***Overview of VB-LoRA**. Left: The model parameters can be represented as a composition of vectors from a vector bank, which is shared across sub-vectors, modules and layers. Right: Architecture of VB-LoRA. We use a top-k softmax function to select k vectors from the vector bank. The selected vectors are then pooled into a sub-vector, which is arranged at a desired position, forming the parameters of LoRA.*

<img src="https://github.com/leo-yangli/VB-LoRA/blob/main/param_comp.png?raw=True" alt="Comparison of the PEFT methods" width="350"/>

***Comparison with other PEFT methods on RoBERTa-Large.*** VB-LoRA achieves higher scores with significantly smaller number of stored parameters.

## Steps to reproduce the results

## NLU
- Modified code for running experiments for Natural Language Understanding experiments.
- Adapted from [LoRA source code](https://github.com/microsoft/LoRA).
#### Create and activate conda env
```console
cd NLU/NLU
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
./scripts_vblora_all/roberta_base_cola.sh
```


## Instruction Tuning

- The code for running Llama2 is adapted from [qlora source code](https://github.com/artidoro/qlora).
- Fine-tuning the Llama2 model requires access to the model weights on HuggingFace. Ensure you have the access before running the code.

#### Create and activate conda env
```console
cd instruction_tuning
conda create -n instruction_tuning python==3.10
conda activate instruction_tuning
```

#### Install the pre-requisites
```console
pip install -r requirements.txt
```

#### Start the experiments
The scripts are located in the "instruction_tuning/scripts" folder.

For example,
```console
cd instruction_tuning
./scripts/finetune_llama2_7b_vblora.sh
```

For evaluation, please use [LLM Judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).

## Math Instruction Tuning
#### Create and activate conda env
```console
cd math_instruction_tuning
conda create -n math_instruction_tuning python==3.8.13
conda activate math_instruction_tuning
```

#### Install the pre-requisites
```console
pip install -r requirements.txt
```

#### Start the experiments
The scripts are located in the "instruction_tuning/scripts" folder.

For example,
```console
./run_instruction_tuning_vblora.sh
```

## Citation
If you found this code useful, please cite our paper.

```  
@inproceedings{li2024vblora,
      title={VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks}, 
      author={Yang Li and Shaobo Han and Shihao Ji},
      booktitle={The 38th Conference on Neural Information Processing Systems (NeurIPS)},
      year={2024}
}
```  
