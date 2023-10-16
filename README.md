# Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation

This is the official repository for ["Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation"](https://arxiv.org/abs/2310.06987), by [Yangsibo Huang](https://hazelsuko07.github.io/yangsibo/), [Samyak Gupta](https://scholar.google.com/citations?user=vuz_F7IAAAAJ&hl=en), [Mengzhou Xia](https://xiamengzhou.github.io/), [Kai Li](https://www.cs.princeton.edu/~li/), and [Danqi Chen](https://www.cs.princeton.edu/~danqic/). 

In this work, we propose the **generation exploitation attack**, an extremely simple approach that disrupts model alignment by only manipulating variations of generation methods. Check out our [project page](https://princeton-sysml.github.io/jailbreak-llm/) for more details and examples!

<p align="center">
  <img src="intro.png" width="100%">
</p>

If you find our implementation and paper helpful, please consider citing our work:

```bibtex
@article{huang2023catastrophic,
  title={Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation},
  author={Huang, Yangsibo and Gupta, Samyak and Xia, Mengzhou and Li, Kai and Chen, Danqi},
  journal={arXiv preprint arXiv:2310.06987},
  year={2023}
}
```

## Table of Contents
- [Updates](https://github.com/Princeton-SysML/Jailbreak_LLM#updates)
- [Installation](https://github.com/Princeton-SysML/Jailbreak_LLM#installation)
- [Preparation](https://github.com/Princeton-SysML/Jailbreak_LLM#preparation)
- [Launch the attack](https://github.com/Princeton-SysML/Jailbreak_LLM#launch-the-attack)
- [Evaluate the attack](https://github.com/Princeton-SysML/Jailbreak_LLM#evaluate-the-attack)
- [Bugs or questions](https://github.com/Princeton-SysML/Jailbreak_LLM#bugs-or-questions)

## Updates
 - **2023-10-11**: We release the first version.

## Installation

Our implementation was based on `transformers==4.32.1` and `torch==2.0.1`.
## Preparation

Our attack is model-agnostic. To launch the attack on a specified model, please download the model and insert/update its path in `configs.py`. The table below provides links to all models we used in the paper.

| Model             | Hugging Face Page                                        |
| ----------------- | ------------------------------------------------------- |
| Vicuna-7B          | [https://huggingface.co/lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)          |
| Vicuna-13B         | [https://huggingface.co/lmsys/vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5)         |
| Vicuna-33B         | [https://huggingface.co/lmsys/vicuna-33b-v1.3](https://huggingface.co/lmsys/vicuna-33b-v1.3)         |
| MPT-7B             | [https://huggingface.co/mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b)                     |
| MPT-30B            | [https://huggingface.co/mosaicml/mpt-30b](https://huggingface.co/mosaicml/mpt-30b)                   |
| Falcon-7B          | [https://huggingface.co/tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b)                   |
| Falcon-40B         | [https://huggingface.co/tiiuae/falcon-40b](https://huggingface.co/tiiuae/falcon-40b)                 |
| LLaMA2-7B          | [https://huggingface.co/meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)    |
| LLaMA2-13B         | [https://huggingface.co/meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf)  |
| LLaMA2-7B-chat     | [https://huggingface.co/meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| LLaMA2-13B-chat    | [https://huggingface.co/meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) |


## Launch the attack

### Default or greedy generation

The commands below runs the default decoding (`temperature=0.1`, `top_p=0.9`) for the `Llama-2-7b-chat-hf` model:
```bash
python attack.py \
    --model Llama-2-7b-chat-hf \
    --use_default \
    --use_system_prompt
```

or the greedy decoding for the `Llama-2-7b-chat-hf` model:
```bash
python attack.py \
    --model Llama-2-7b-chat-hf \
    --use_greedy \
    --use_system_prompt
```

### Exploited generation

The command below exploits various decoding settings for the `Llama-2-7b-chat-hf` model (with the system prompt disabled):
```bash
python attack.py \
    --model Llama-2-7b-chat-hf \
    --tune_temp \
    --tune_topp \
    --tune_topk \
    --n_sample 1
```

You can increase the `--n_sample` parameter to generate more examples for each prompt, which potentially makes the attack stronger.

## Evaluate the attack

Before running the evaluation, please make sure that you have prepared a risk scorer model (saved in `SCORER_PATH`) for the attacker to pick the best attack result among several candidates. In the meantime, you need to prepare a separate evaluator (saved in `EVALUATOR_PATH`) for the final scoring purpose. 

You may download our scorer and evaluator [here](https://drive.google.com/drive/folders/1G0bxoe7lNfpaEQKqBb3JW5M0db9GJ5BR?usp=sharing).

To evaluate the attack with substring match by [Zou et al.](https://arxiv.org/abs/2307.15043), you can use the following command:
```bash
python evaluate.py \
    --model Llama-2-7b-chat-hf \
    --config 'exploited' \
    --matching_only \
    --scorer_path SCORER_PATH
```

or using our evaluator:
```bash
python evaluate.py \
    --model Llama-2-7b-chat-hf \
    --config 'exploited' \
    --evaluator_path EVALUATOR_PATH \
    --scorer_path SCORER_PATH
```

which summarizes results in a json file. For instance, here is the output for `Llama-2-7b-chat-hf` model with `n_sample=1`. 
```python
{
    "best_attack_config": {
        "temp": {
            "temp_0.95": "25"   # Most vulnerable temp is 0.95, which gives 25% ASR
        },
        "topk": {
            "topk_500": "26"   # Most vulnerable top-k is 500, which gives 26% ASR
        },
        "topp": {
            "topp_0.7": "29"   # Most vulnerable temp is 0.7, which gives 29% ASR
        }
    },
    "greedy": "16",            # Greedy decoding without system prompt gives 16% ASR
    "break_by_temp": "47",     # Exploiting temp only gives 47% ASR
    "break_by_topk": "54",     # Exploiting top-k only gives 54% ASR
    "break_by_topp": "77",     # Exploiting top-p only gives 77% ASR
    "break_by_all": "81"       # Exploiting all decoding strategies gives 81% ASR
}
``` 

You can aggregate results for different models (to obtain Table 1 in our paper) using the example shown in `aggregate_results.ipynb`.


## Bugs or questions
If you have any questions related to the code or the paper, feel free to email Yangsibo (yangsibo@princeton.edu) or open an issue. Please try to specify the problem with details so we can help you better and quicker!
