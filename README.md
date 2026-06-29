
<div align="center">

# 🏴‍☠️ IMAGINE Hackathon 2026
*⚡ How much energy does it really take to train a competent vision model? ⚡*
</div>

In this repo, we train a **ViT-S/16** for **image classification** on the **ImageNet-1k** dataset.

The goal is to achieve **85% top-5 accuracy** on our test set using as little energy as possible.

Our baseline trains for under 7 hours with a single A6000 GPU and 64GB of system RAM, consuming around 2630 watt-hours. The baseline is already partially optimized with Flash Attention, mixed precision and model compilation, but we can for sure do better!

### Index
- [⚙️ Before you Start](#️-before-you-start)
    - [Dataset](#dataset)
    - [uv](#uv)
    - [Weights & Biases](#weights--biases)
    - [CodeCarbon](#codecarbon)
        - [Electricity Maps [Optional]](#electricity-maps-optional)
- [📁 Project Structure](#-project-structure)
- [🧪 Defining Experiments](#-defining-experiments)
    - [Experiment Tagging](#experiment-tagging)
    - [Hyperparameter Search](#hyperparameter-search)
- [🚀 Training](#-training)
    - [⚠️ IMPORTANT](#️-important)
- [🔄 Validating](#-validating)
- [✅ Testing](#-testing)

## ⚙️ Before you Start
We need to set up some things before we start training. First clone the repo and `cd` into it, then do the following:

### Dataset
To download our split of ImageNet-1k, run `download_imagenet.sh`:
```bash
./download_imagenet.sh
```
After the download finishes, you should find the `train` and `val` partitions under the `data/` folder.

Alternatively, download the official ImageNet-1k train split (you might need to create an account on their [website](https://www.image-net.org/login.php) first):
```bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
```
If the command doesn't work, download the train split manually from [here](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php#images) - Training images (Task 1 & 2).

Then extract the contents and locate `Data/CLS-LOC/train`. Place it under `data/`, as `data/train`. Then, `cd` into `split_train_set`, check that the paths and options in `split_val_by_spec.py` are correct, and run `python split_val_by_spec.py`.

### uv
We are going to use the [uv package manager](https://docs.astral.sh/uv/). To install it, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or if you don't have `curl`:
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

Before moving on to the next steps, it is a good idea to get the environment ready:
```bash
uv venv
uv pip install -r pyproject.toml
```

### Weights & Biases
We are going to log all of our experiments to the [`imaginelab`](https://wandb.ai/imaginelab) Weights & Biases dashboard. Create an account if you don't have one, and ask someone to add you if you are not a member.

Make sure you are logged in before launching experiments!
```bash
uv run wandb login
```

### CodeCarbon
We are going to use CodeCarbon to measure the energy consumption and carbon emissions of our training runs.
CodeCarbon has an API that we can use to upload all the measurements to their dashboard, but to do this we all need to have an account and that CodeCarbon can authenticate to the API. Step by step:
1) Go to https://dashboard.codecarbon.io/ > `Sign in or create an account`
2) Click `Register` at the bottom, fill and submit the form with your ENPC/uni email
3) Send me (Marta López) your email address on Slack so that I can add you to our organization
4) On your machine, install and configure CodeCarbon

   **IMPORTANT**: Forward port 8090 either in your IDE or when connecting to your server through SSH: `ssh -L 8090:localhost:8090 <IMAGINE server name>`, then run:
    ```bash
    export BROWSER=echo
    uv run codecarbon login
    # Open the link in a browser and log in
    uv run codecarbon config
    # Select:
    #    organization -> IMAGINE
    #    project -> IMAGINE Hackathon 2026
    #    experiment -> Baseline
    ```
6) To get CPU power consumption measurements, run:
    ```bash
    sudo chmod -R a+r /sys/class/powercap/intel-rapl
    ```

> You can check out the total energy consumption of the hackathon at the [dashboard](https://dashboard.codecarbon.io/)!

#### Electricity Maps [Optional]
To get more accurate carbon emission measurements, CodeCarbon can use the ElectricityMaps API. For this to work, follow these steps:
1) Go to https://app.electricitymaps.com/auth/signup
2) Enter your ENPC/uni email and click `Sign up`, then follow the instruction to complete your registration
3) Scroll down to the `Resources` section > `API Key`, create a new API key and copy it to your clipboard
   
   Alternatively, go to `Developer Hub` (`</>` icon on the left sidebar) > `Playground`, and copy the key from `-H "auth-token: <your API key>"`
4) Write the API key to `electricity_maps_key.txt` in the root directory of the repo:
    ```bash
    echo <your API key> >> ./electricity_maps_key.txt
    ```


## 📁 Project Structure
This repo is based on the [Lightning + Hydra template by ashleve](https://github.com/ashleve/lightning-hydra-template).

There are three main components that you can play around with:
- The [`datamodule`](./configs/datamodule/) reads the raw data, applies data augmentation, collates batches, and sends them to the GPU.
- The [`module`](./configs/module/) defines the network, schedulers, and optimizers and controls what happens in each training step.
- The [`trainer`](./configs/trainer/) configures some global training options, such as the total number of epochs, gradient clipping settings, or the use of mixed precision.

So you will be working mainly with those three configs, as well as [`experiment`](./configs/experiment/), and maybe [`hparams_search`](./configs/hparams_search) and [`callbacks`](./configs/callbacks/).

## 🧪 Defining Experiments
To define a new experiment, create a YAML file under [`configs/experiment`](./configs/experiment).

If needed, override the `module`, `datamodule` or `trainer` configs as done in the [example](configs/experiment/example.yaml) with your own config files defined in the [`module/`](./configs/module/), [`datamodule/`](./configs/datamodule/) or [`trainer/`](./configs/trainer/) directories under [`configs/`](./configs/), respectively.

You may also define new versions of [`imagenet_datamodule.py`](./src/datamodules/imagenet_datamodule.py) and [`imagenet_module.py`](./src/modules/imagenet_module.py) in the corresponding directories under [`src/`](./src/). You can define new network architectures under [`src/modules/nets/`](./src/modules/nets/).

### Experiment Tagging
To keep things tidy, please define your `team_name` in the [main train config](./configs/train.yaml), since it should be the same across all experiments. Then, for every experiment you create, remember to define `experiment_name` and `tags` in the experiment config.
- `team_name`: name of your team. Can be a single letter, the team leader's name, or a short name, as long as it is the same for all team members.
- `experiment_name`: name of the approach you are trying out. Experiment logs are timestamped, so running the same experiment several times will not overwrite previous output.
- `tags`: a *project stage* and a *run type* tag. The available options are:
    - Project stage options: `data`, `explore`, `baseline`, `ablate`, `hyperparam`, `final`, `<custom tag>`
    - Run type options: `train`, `pre-train`, `post-train`, `debug`, `<custom tag>`

### Hyperparameter Search
You can use the Hydra `--multirun` (or `-m`) option to launch a simple, *sequential* grid search as shown in the [documentation](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/). For example:

```bash
uv run src/train.py --multirun datamodule.batch_size=32,64,128 module.optimizer.lr="range(0.01,0.06,0.01)" tags="[hyperparam,train]"
```

You can also try Optuna for a smarter hyperparameter search. There is an example config in [optuna.yaml](./configs/hparams_search/optuna.yaml), you can use it by indicating `hparams_search: optuna` in your experiment config.

> Don't forget to use the `hyperparam` tag in the tags field of the config!

## 🚀 Training
Run:
```bash
uv run src/train.py
```
with any Hydra command-line overrides that you need for your experiment.

### ⚠️ IMPORTANT
> We are also measuring **CPU** and **RAM** consumption, not just GPU consumption. CodeCarbon tries to isolate the consumption of the process that it is tracking, but you will get more reliable measurements if you **don't run experiments in parallel** on the same machine.

> Energy consumption is typically the same across epochs, so you can **train for a couple epochs** only to compare different approaches.

> You can visualize the GPU, CPU, and RAM power consumption measured by CodeCarbon on **wandb**. 

## 🔄 Validating

You can evaluate your ideas on the validation set before we release the official test set. Complete the [eval config](./configs/eval.yaml) with your selected **checkpoints** and your **team name**, then run:
```bash
uv run src/valid.py
```
This will send the energy consumption and metrics to a centralized evaluation server.

**IMPORTANT**: For the upload to work, your machine needs to be on the lab network. If you trained elsewhere, copy your checkpoints to a lab server and run the script there.

## ✅ Testing
Once the test set has been released, modify the tags on the [eval config](./configs/eval.yaml) to `['final', 'evaluate']`, then run:
```bash
uv run src/test.py
```
This will register the output of the model for each image in the test set and upload the results and CodeCarbon metrics to the evaluation server.

**IMPORTANT**: For the upload to work, your machine needs to be on the lab network. If you trained elsewhere, copy your checkpoints to a lab server and run the script there.

We will reveal the test performance after everyone has submitted their results!
