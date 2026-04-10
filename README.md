# PointFlowMatch: Learning Robotic Manipulation Policies from Point Clouds with Conditional Flow Matching

Repository providing the source code for the paper "Learning Robotic Manipulation Policies from Point Clouds with Conditional Flow Matching", see the [project website](http://pointflowmatch.cs.uni-freiburg.de/). Please cite the paper as follows:

	@article{chisari2024learning,
	  title={Learning Robotic Manipulation Policies from Point Clouds with Conditional Flow Matching},
      shorttile={PointFlowMatch},
	  author={Chisari, Eugenio and Heppert, Nick and Argus, Max and Welschehold, Tim and Brox, Thomas and Valada, Abhinav},
	  journal={Conference on Robot Learning (CoRL)},
	  year={2024}
	}

## Installation

- Add env variables to your `.bashrc`

```bash
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

- Install dependencies

```bash
conda create --name pfp_env python=3.10
conda activate pfp_env
bash bash/install_deps.sh
bash bash/install_rlbench.sh

# Get diffusion_policy from my branch
cd ..
git clone git@github.com:chisarie/diffusion_policy.git && cd diffusion_policy && git checkout develop/eugenio 
pip install -e ../diffusion_policy

# 3dp install
cd ..
git clone git@github.com:YanjieZe/3D-Diffusion-Policy.git && cd 3D-Diffusion-Policy
cd 3D-Diffusion-Policy && pip install -e . && cd ..

# If locally (doesnt work on Ubuntu18):
pip install rerun-sdk==0.15.1
```

## Pretrained Weights Download

Here you can find the pretrained checkpoints of our PointFlowMatch policies for different RLBench environments. Download and unzip them in the `ckpt` folder.

| unplug charger | close door | open box | open fridge | frame hanger | open oven | books on shelf | shoes out of box |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [1717446544-didactic-woodpecker](http://pointflowmatch.cs.uni-freiburg.de/download/1717446544-didactic-woodpecker.zip) | [1717446607-uppish-grebe](http://pointflowmatch.cs.uni-freiburg.de/download/1717446607-uppish-grebe.zip) | [1717446558-qualified-finch](http://pointflowmatch.cs.uni-freiburg.de/download/1717446558-qualified-finch.zip) | [1717446565-astute-stingray](http://pointflowmatch.cs.uni-freiburg.de/download/1717446565-astute-stingray.zip) | [1717446708-analytic-cuckoo](http://pointflowmatch.cs.uni-freiburg.de/download/1717446708-analytic-cuckoo.zip) | [1717446706-natural-scallop](http://pointflowmatch.cs.uni-freiburg.de/download/1717446706-natural-scallop.zip) | [1717446594-astute-panda](http://pointflowmatch.cs.uni-freiburg.de/download/1717446594-astute-panda.zip) | [1717447341-indigo-quokka](http://pointflowmatch.cs.uni-freiburg.de/download/1717447341-indigo-quokka.zip) |

## Evaluation

To reproduce the results from the paper, run:

```bash
python scripts/evaluate.py log_wandb=True env_runner.env_config.vis=False policy.ckpt_name=<ckpt_name>
```

Where `<ckpt_name>` is the folder name of the selected checkpoint. Each checkpoint will be automatically evaluated on the correct environment.

## Training

To train your own policies instead of using the pretrained checkpoints, you first need to collect demonstrations:

```bash
bash bash/collect_data.sh
```

Then, you can train your own policies:

```bash
python scripts/train.py log_wandb=True dataloader.num_workers=8 task_name=<task_name> +experiment=<experiment_name>
```

Valid task names are all those supported by RLBench. In this work, we used the following tasks: `unplug_charger`, `close_door`, `open_box`, `open_fridge`, `take_frame_off_hanger`, `open_oven`, `put_books_on_bookshelf`, `take_shoes_out_of_box`.

Valid experiment names are the following, and they represent the different baselines we tested: `adaflow`, `diffusion_policy`, `dp3`, `pointflowmatch`, `pointflowmatch_images`, `pointflowmatch_ddim`, `pointflowmatch_so3`.