import wandb
import sys
import pandas as pd

def check_if_experiment_exists(project, entity, configs, keys_of_interest):
    prefixed_configs = {
        f"config.{key_of_interest}": configs[key_of_interest] for key_of_interest in keys_of_interest}
    api = wandb.Api()
    old_runs = list(api.runs(path=f"{entity}/{project}",
            filters={
                "state": {"$in": ["finished", "running"]},
                **prefixed_configs,
            }))

    if len(old_runs) > 0:
        for run in old_runs:
            if run.state == "finished":
                print("Max epoch: ", pd.DataFrame(run.scan_history())["epoch"].max())
                if pd.DataFrame(run.scan_history())["epoch"].max() >= 99:
                    print("Run exists")
                    sys.exit()
            if run.state == "running":
                print("Run exists")
                sys.exit()
    return False

def setup_wandb(project, configs, **kwargs):
    run = wandb.init(
            project=project,           
            config=configs,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,            
            settings=wandb.Settings(start_method='fork'),
            **kwargs,
        )

def maybe_setup_wandb(mode, project, configs,
        keys_of_interest=["env", "sampler", "seed"],
        **kwargs):
    if check_if_experiment_exists(project, 
            configs,
            keys_of_interest=keys_of_interest):
        print("Run exists")
        sys.exit()
    setup_wandb(project, 
        entity, 
        configs, 
        mode=mode)


def get_d3rlpy_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper-medium-v2')
    parser.add_argument('--sampler', type=str, default='uniform')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--wandb', type=str, default=None)
    parser.add_argument('--project', type=str, default="offline-subopt-td3bc")
    args = parser.parse_args()
    return args