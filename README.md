# FlowerTune LLM on General NLP Dataset

This directory conducts federated instruction tuning with a pretrained [Gemma2-9b-cpt-sahabatai-v1-instruct](https://huggingface.co/GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct) model on a [General NLP dataset](https://huggingface.co/datasets/vicgalle/alpaca-gpt4).
We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.
Flower's Simulation Engine is used to simulate the LLM fine-tuning process in federated way,
which allows users to perform the training on a single GPU.


## Methodology

In our implementation, we use [Gemma2-9b-cpt-sahabatai-v1-instruct](https://huggingface.co/GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct) as the pre-trained model and modify the training settings compared to the baseline, which uses the `Mistral-7B-v0.3` model. Additionally, we leverage the [FlexLoRA](https://arxiv.org/abs/2402.11505) to reduce the LoRA aggregation noise. 

|Setting                                                | Baseline                 | Ours                                            |
| --                                                    |  --                      |  --                                             |
|model.name                                             |mistralai/Mistral-7B-v0.3 | GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct |
|model.lora.peft-lora-r                                 | 32                       | 8                                               |
|model.lora.peft-lora-alpha                             | 64                       | 16                                              |
|model.lora.peft-lora-alpha                             | 64                       | 16                                              |
|train.training-arguments.per-device-train-batch-size   | 16                       | 2                                               |
|train.training-arguments.gradient-accumulation-steps   | 1                        | 4                                               |
|train.training-arguments.logging-steps                 | 10                       | 1                                               |
|num-server-rounds                                      | 200                      | 10                                              |
|options.backend.client-resources.num-cpus              | 6                        | 2                                               |
|use_flexlora                                           | null                     | 0 for FedAvg (w/o FlexLoRA), 1 for w FlexLoRA   |

The model weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/10FD_6PO4YlMOHmzRxHl-IsiH_0drb0Gc?usp=sharing).

## Environments setup

Project dependencies are defined in `pyproject.toml`. Install them in an activated Python environment with:

```shell
pip install -e .
```

## Experimental setup

The dataset is divided into 20 partitions in an IID fashion, a partition is assigned to each ClientApp.
We randomly sample a fraction (0.1) of the total nodes to participate in each round, for a total of `10` rounds.
All settings are defined in `pyproject.toml`.

> [!IMPORTANT]
> Please note that `[tool.flwr.app.config.static]` and `options.num-supernodes` under `[tool.flwr.federations.local-simulation]` are not allowed to be modified for fair competition if you plan to participated in the [LLM leaderboard](https://flower.ai/benchmarks/llm-leaderboard).


## Running the challenge

[Gemma2-9b-cpt-sahabatai-v1-instruct](https://huggingface.co/GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct) is freely accessible and does not require special access. Simply follow the instructions [here](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command) to log in your account. Note you only need to complete this stage once in your development machine:

```bash
huggingface-cli login
```

Run the challenge with default config values.
The configs are defined in `[tool.flwr.app.config]` entry of `pyproject.toml`, and are loaded automatically.

```bash
# To test out the Gemma2-9b-cpt-sahabatai-v1-instruct model with FlexLoRA, set `use_flexlora` to 1 in `pyproject.toml`
# otherwise, pls set `use_flexlora` to 0.
flwr run
```

## VRAM consumption

We use the [Gemma2-9b-cpt-sahabatai-v1-instruct](https://huggingface.co/GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct) model with 4-bit quantization by default. The estimated overall VRAM consumption for GeneralNLP challenge is shown below:

The peak VRAM consumption for our implementation using FedAvg is:

| Challenges | GeneralNLP | 
| :--------: | :--------: |
|    VRAM    | ~27.59 GB  | 

Using FlexLoRA to aggregate model updates requires higher peak VRAM to store intermediate full-rank matrices. We leave memory optimization for future work.  The peak VRAM consumption for our implementation using FlexLoRA is:

| Challenges | GeneralNLP | 
| :--------: | :--------: |
|    VRAM    | ~33.33 GB  | 

You can adjust the CPU/GPU resources you assign to each of the clients based on your device, which are specified with `options.backend.client-resources.num-cpus` and `options.backend.client-resources.num-gpus` under `[tool.flwr.federations.local-simulation]` entry in `pyproject.toml`.


## Model saving

The global PEFT model checkpoints are saved every 5 rounds after aggregation on the sever side as default, which can be specified with `train.save-every-round` under [tool.flwr.app.config] entry in `pyproject.toml`.

> [!NOTE]
> Please provide the last PEFT checkpoint if you plan to participated in the [LLM leaderboard](https://flower.ai/benchmarks/llm-leaderboard).

## Troubleshooting

```
TypeError: not all arguments converted during string formatting
...
Message: 'You have set `use_cache` to `False`, but cache_implementation is set to hybrid. cache_implementation will have no effect.' Arguments: (<class 'UserWarning'>,)
```

This warning is located in `"transformers/generation/configuration_utils.py"` line 789. The `validate()` function in `GenerationConfig` ensures that all parameters are correctly set and do not conflict with each other. If `use_cache=False`, but the user still sets cache-related parameters (e.g., `cache_implementation`, `cache_config`), the warning alerts them that those settings will be ignored. The developers explicitly chose to issue a warning instead of raising an error to avoid forcing users to change their configurations.

However, during the training, `use_cache=False` is commonly used to save memory. Even if all other cache-related parameters are unset in advance, this warning still appears because it only checks whether use_cache is set to False, rather than verifying if other cache-related settings are also disabled.

One way to remove the `TypeError` is to remove the `UserWarning` from this line, then it will only output a notification.

```python
for arg_name in ("cache_implementation", "cache_config", "return_legacy_cache"):
    if getattr(self, arg_name) is not None:
        logger.warning_once(
            no_cache_warning.format(cache_arg=arg_name, cache_arg_value=getattr(self, arg_name)),
            UserWarning,  % Remove
        )
```

```python
for arg_name in ("cache_implementation", "cache_config", "return_legacy_cache"):
    if getattr(self, arg_name) is not None:
        logger.warning_once(
            no_cache_warning.format(cache_arg=arg_name, cache_arg_value=getattr(self, arg_name)),
        )
```