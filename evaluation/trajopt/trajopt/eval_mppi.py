"""
This is a eval script for launching mjrl training using hydra
"""
import numpy as np
import os
import time as timer
import glob
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import pickle 
from tqdm import tqdm 
import multiprocessing as mp
from matplotlib import pyplot as plt
from moviepy.editor import ImageSequenceClip
import skvideo.io
import os 
import json
import wandb
from PIL import Image 
 
import sys
sys.path.append('/home/puhao/dev/MAH/CodeRepo/vip/evaluation/trajopt/trajopt/')
from mj_envs.envs.env_variants import register_env_variant
from trajopt.envs.obs_wrappers import env_constructor
from trajopt.algos.mppi import MPPI
from icecream import install
install()

@hydra.main(config_name="eval_mppi_config", config_path="config")
def main_worker(cfg: DictConfig):
    print(os.getcwd())
    # use assert to garantee if embedding_reward is true, then env_kwargs.embedding_reward is true
    assert cfg.embedding_reward ==  True
    assert cfg.env_kwargs.embedding_reward == True 

    # makedirs for cfg.exp_dir
    os.makedirs(cfg.exp_dir, exist_ok=True)
    os.makedirs(cfg.vis_dir, exist_ok=True)

    cfg.env_kwargs.load_path = cfg.embedding
    cfg.job_name = f'{cfg.env}#{cfg.embedding}#{cfg.camera}#{cfg.task_type}'

    # activate cfg.slurm if run on slurm
    if os.environ.get('SLURM') is not None:
        cfg.slurm = True

    ## set output logger and tensorboard
    if cfg.slurm:
        logger.remove(handler_id=0) # remove default handler
    logger.add(cfg.exp_dir + '/runtime.log')
    
    logger.info('Configuration: \n' + OmegaConf.to_yaml(cfg))
    
    # read the init_timestep and goal_timestep from ../dataset based on cfg.env_kwargs.task_type
    task_config_path = './evaluation/dataset/' + cfg.env + '/0/task_config.json'
    with open(task_config_path, 'r') as f:
        task_config = json.load(f)
    
    # set init_timestep and goal_timestep
    cfg.env_kwargs.init_timestep = task_config[cfg.task_type]['init_timestep']
    cfg.env_kwargs.goal_timestep = task_config[cfg.task_type]['goal_timestep']
    logger.info(f"task type: {cfg.task_type}")
    logger.info(f"init_timestep: {cfg.env_kwargs.init_timestep}, goal_timestep: {cfg.env_kwargs.goal_timestep}")

    # get env_kwargs from cfg
    env_kwargs = cfg.env_kwargs
    OmegaConf.set_struct(env_kwargs, False)
    # construct env
    if cfg.slurm:
        env_kwargs.env_xml_path = env_kwargs['env_xml_path_slurm']
        env_kwargs.env_cfg_path = env_kwargs['env_cfg_path_slurm']
        env_kwargs.demo_basedir = env_kwargs['demo_basedir_slurm']
    else:
        env_kwargs.env_xml_path = env_kwargs['env_xml_path_local']
        env_kwargs.env_cfg_path = env_kwargs['env_cfg_path_local']
        env_kwargs.demo_basedir = env_kwargs['demo_basedir_local']
    del env_kwargs['env_xml_path_slurm']
    del env_kwargs['env_cfg_path_slurm']
    del env_kwargs['env_xml_path_local']
    del env_kwargs['env_cfg_path_local']
    del env_kwargs['demo_basedir_local']
    del env_kwargs['demo_basedir_slurm']
    OmegaConf.set_struct(env_kwargs, True)
    env = env_constructor(**env_kwargs)

    # get mean and sigma from cfg
    mean = np.zeros(env.action_dim)
    sigma = 1.0*np.ones(env.action_dim)
    filter_coefs = [sigma, cfg.filter.beta_0, cfg.filter.beta_1, cfg.filter.beta_2]

    # Generate trajectories with seed
    runs_success = []
    for i_run in range(len(cfg.seed)):
        start_time = timer.time()
        logger.info(f"Run {i_run} with seed {cfg.seed[i_run]}")

        # set np and env seed
        np.random.seed(cfg.seed[i_run])
        env.reset(seed=cfg.seed[i_run])

        # reset mppi agent
        agent = MPPI(env,
                     agent_ago=cfg.env_kwargs.agent_ago,
                     H=cfg['plan_horizon'],
                     paths_per_cpu=cfg['paths_per_cpu'],
                     num_cpu=cfg['num_cpu'],
                     kappa=cfg['kappa'],
                     gamma=cfg['gamma'],
                     mean=mean,
                     filter_coefs=filter_coefs,
                     default_act=cfg['default_act'],
                     seed=cfg.seed[i_run],
                     env_kwargs=env_kwargs)
        
        # trajectory optimization
        distances = {}
        for camera in agent.env.env.cameras:
            distances[camera] = []
            goal_embedding = agent.env.env.goal_embedding[camera]
            distance = np.linalg.norm(agent.sol_embedding[-1][camera]-goal_embedding)
            distances[camera].append(distance)
        
        for i_timestep in tqdm(range(cfg.H_total)):
            agent.train_step(cfg['num_iter'])
            step_info = agent.sol_info[-1]
            logger.info(f"step {i_timestep} | solved {step_info['solved']}, rwd_solved {step_info['rwd_solved']}, done {step_info['done']}")
        
        logger.info(f"Run {i_run} with seed {cfg.seed[i_run]} finished in {timer.time() - start_time:.2f} seconds")
        logger.info(f"task success: {step_info['solved']}")
        
        # save the gif
        if cfg.save_gifs:
            video_path = os.path.join(cfg.vis_dir, f'{cfg.job_name}#{i_run}.gif')
            frames = agent.animate_result_offscreen(camera_name=camera)
            cl = ImageSequenceClip(frames, fps=24)
            cl.write_gif(video_path, fps=24)
            logger.info(f"save video to {video_path}")
        
        # save succ or fail to runs_success list
        runs_success.append(step_info['solved'])
    
    # save the success rate to output folder
    success_rate = np.mean(runs_success)
    logger.info(f"success rate: {success_rate}")
    with open(os.path.join(cfg.exp_dir, 'results.json'), 'w') as f:
        json.dump({'success_rate': f"{success_rate:.2f}", 
                   'succ/runs': f"{int(np.sum(runs_success))}/{len(runs_success)}",
                   'runs_results': {i_run: str(runs_success[i_run]) for i_run in range(len(runs_success))}}, f)


# run main function
if __name__ == "__main__":
    mp.set_start_method('spawn')
    main_worker()
