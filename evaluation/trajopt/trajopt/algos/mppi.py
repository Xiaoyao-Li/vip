"""
This implements a shooting trajectory optimization algorithm.
The closest known algorithm is perhaps MPPI and hence we stick to that terminology.
Uses a filtered action sequence to generate smooth motions.
"""
import os
import numpy as np
import torch
from mjrl.utils.logger import DataLog

from trajopt.algos.trajopt_base import Trajectory
from trajopt.utils.utils import gather_paths_parallel

from PIL import Image
from omegaconf import OmegaConf
from loguru import logger

from aamzoo.seggpt.interface import seggpt_prepare_model, seggpt_model_inference, seggpt_preprocess_image
from aamzoo.e2fgvi.interface import e2fgvi_prepare_model, e2fgvi_preprocess_image, e2fgvi_model_inference

class MPPI(Trajectory):
    def __init__(self, env, agent_ago, H, paths_per_cpu,
                 num_cpu=1,
                 kappa=1.0,
                 gamma=1.0,
                 mean=None,
                 filter_coefs=None,
                 default_act='repeat',
                 warmstart=True,
                 seed=123,
                 agent_ago_dict = dict(
                    seggpt_prompt_dir='/home/puhao/dev/MAH/CodeRepo/vip/evaluation/dataset/seggpt_prompt',
                    e2fgvi_ckpt_path = '/home/puhao/dev/MAH/DataPreprocess/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth',
                    e2fgvi_model_name = 'e2fgvi_hq',
                    seggpt_ckpt_path = '/home/puhao/dev/MAH/CodeRepo/Painter/SegGPT/SegGPT_inference/seggpt_vit_large.pth',
                    seggpt_model_name = 'seggpt_vit_large_patch16_input896x448',),
                 env_kwargs=None):
        self.agent_ago = agent_ago
        self.agent_ago_cfg = OmegaConf.create(agent_ago_dict)

        self.env, self.seed = env, seed
        self.env_kwargs = env_kwargs
        self.n, self.m = env.observation_dim, env.action_dim
        self.H, self.paths_per_cpu, self.num_cpu = H, paths_per_cpu, num_cpu
        self.warmstart = warmstart

        self.logger = DataLog() 
        
        self.mean, self.filter_coefs, self.kappa, self.gamma = mean, filter_coefs, kappa, gamma
        if mean is None:
            self.mean = np.zeros(self.m)
        if filter_coefs is None:
            self.filter_coefs = [np.ones(self.m), 1.0, 0.0, 0.0]
        self.default_act = default_act

        self.sol_state = []
        self.sol_act = []
        self.sol_reward = []
        self.sol_obs = []
        self.sol_embedding = [] 
        self.sol_info = [] 

        self.env.reset()
        self.env.set_seed(seed)
        self.env.reset(seed=seed)
        self.sol_state.append(self.env.get_env_state().copy())
        self.sol_obs.append(self.env.get_obs())
        self.act_sequence = np.ones((self.H, self.m)) * self.mean
        self.init_act_sequence = self.act_sequence.copy()

        if env_kwargs['embedding_reward']:
            self.sol_embedding.append(self.env.env.get_views(embedding=True))
        self._prepare_agentago()

    def update(self, paths):
        num_traj = len(paths)
        act = np.array([paths[i]["actions"] for i in range(num_traj)])
        R = self.score_trajectory(paths)
        S = np.exp(self.kappa*(R-np.max(R)))

        # blend the action sequence
        weighted_seq = S*act.T
        act_sequence = np.sum(weighted_seq.T, axis=0)/(np.sum(S) + 1e-6)
        self.act_sequence = act_sequence

    def advance_time(self, act_sequence=None):
        act_sequence = self.act_sequence if act_sequence is None else act_sequence
        # accept first action and step
        action = act_sequence[0].copy()
        self.env.real_env_step(True)
        s, r, _, info, i_cam = self.env.step(action)

        self.sol_act.append(action)
        self.sol_state.append(self.env.get_env_state().copy())
        self.sol_obs.append(self.env.get_obs())
        self.sol_reward.append(r)
        self.sol_info.append(info)

        if 'obs_embedding' in info:
            self.sol_embedding.append(self.env.env.get_views(embedding=True))    
        
        # get updated action sequence
        if self.warmstart:
            self.act_sequence[:-1] = act_sequence[1:]
            if self.default_act == 'repeat':
                self.act_sequence[-1] = self.act_sequence[-2]
            else:
                self.act_sequence[-1] = self.mean.copy()
        else:
            self.act_sequence = self.init_act_sequence.copy()

    def score_trajectory(self, paths):
        scores = np.zeros(len(paths))
        
        for i in range(len(paths)):
            scores[i] = paths[i]["post_rewards"][-1] # -V(s_T;g)
            # scores[i] = paths[i]["rewards"][-1] # -V(s_T;g)
        return scores

    def do_rollouts(self, seed):
        paths = gather_paths_parallel(self.env,
                                      self.sol_state[-1],
                                      self.act_sequence,
                                      self.filter_coefs,
                                      seed,
                                      self.paths_per_cpu,
                                      self.num_cpu,
                                      env_kwargs=self.env_kwargs
                                      )
        return paths

    def train_step(self, niter=1):
        t = len(self.sol_state) - 1
        for _ in range(niter):
            paths = self.do_rollouts(self.seed + t)
            # add imgs_camera to paths dict
            # TODO: compute reward from imgs_camera with env.embedding
            paths = self._compute_reward_with_batch(paths)
            self.update(paths)
        self.advance_time()

    def _prepare_agentago(self):
        if not self.agent_ago:
            logger.info(f'agentago is **{self.agent_ago}**, skipping agentago model preparation')
            return
        logger.info(f'agentago is **{self.agent_ago}**, preparing agentago model')
        device = self.env.env.device
        assert len(self.env.env.cameras) == 1
        camera = self.env.env.cameras[0]
        ref_img = Image.open(os.path.join(self.agent_ago_cfg.seggpt_prompt_dir, f'{camera}_image.png')).convert('RGB')
        ref_tgt = Image.open(os.path.join(self.agent_ago_cfg.seggpt_prompt_dir, f'{camera}_mask.png')).convert('RGB')

        seggpt_ref = dict(image=ref_img, target=ref_tgt)
        self.agentago_ref = seggpt_ref

        self.agentago_model = dict()
        self.agentago_model['seggpt'] = seggpt_prepare_model(self.agent_ago_cfg.seggpt_ckpt_path, 
                                                                   arch=self.agent_ago_cfg.seggpt_model_name).to(device)
        self.agentago_model['e2fgvi'] = e2fgvi_prepare_model(self.agent_ago_cfg.e2fgvi_ckpt_path,
                                                                   model=self.agent_ago_cfg.e2fgvi_model_name).to(device)

    def _process_to_agentago_imgs(self, imgs_camera, seggpt_bs=8, e2fgvi_bs=8):
        """
        input
            imgs_camera: (B, L (usually is 12), C, H, W), uint8
        output
            last_imgs_camera: (B, T (reset to 1), C, H, W), uint8
        """
        device = self.env.env.device
        if not self.agent_ago:
            return imgs_camera[:, -1:]
        last_imgs_camera = imgs_camera[:, -1]
        last_imgs_camera = np.transpose(last_imgs_camera, (0, 2, 3, 1)).astype('uint8')
        assert last_imgs_camera.shape[0] % seggpt_bs == 0, f'batch size {last_imgs_camera.shape[0]} not divisible by seggpt batch size {seggpt_bs}'
        assert last_imgs_camera.shape[0] % e2fgvi_bs == 0, f'batch size {last_imgs_camera.shape[0]} not divisible by e2fgvi batch size {e2fgvi_bs}'

        # do seggpt
        image_batch, target_batch, ori_size = seggpt_preprocess_image(last_imgs_camera, self.agentago_ref['image'], self.agentago_ref['target'])
        seggpt_output = []
        for i in range(0, image_batch.shape[0], seggpt_bs):
            # logger.info(f'Running seggpt batch {i} to {i+seggpt_bs}')
            item_seggpt_output = seggpt_model_inference(image_batch[i:i+seggpt_bs], target_batch[i:i+seggpt_bs],
                                                        ori_size, self.agentago_model['seggpt'], device)
            seggpt_output.append(item_seggpt_output)
        torch.cuda.empty_cache()
        seggpt_output = np.concatenate(seggpt_output, axis=0)

        # do e2fgvi
        seggpt_output /= 255.
        last_imgs_camera = last_imgs_camera.astype("float32") / 255.
        
        # preprocess image
        masked_frames, length_frames_per_ins = e2fgvi_preprocess_image(last_imgs_camera, seggpt_output, ori_size)
        masked_frames = np.transpose(masked_frames, (1, 0, 2, 3, 4))
        length_frames_per_ins = 1
        pred_frames = e2fgvi_model_inference(masked_frames, length_frames_per_ins, ori_size, self.agentago_model['e2fgvi'], device)
        # pred_frames: (B, L[usually is 1], H, W, C)
        pred_frames = (pred_frames * 255.).astype('float32')

        # image showcase to check
        # Image.fromarray(np.transpose(pred_frames[0,0], (0, 1, 2)).astype('uint8'), 'RGB').show()
        
        pred_frames = np.transpose(pred_frames, (0, 1, 4, 2, 3))
        return pred_frames


    def _compute_reward_with_batch(self, paths):
        # batch the imgs_camera and compute embeddings
        imgs_camera = np.stack([paths[i]["imgs_camera"] for i in range(len(paths))], axis=0)

        # TODO: process agentago paths
        imgs_camera = self._process_to_agentago_imgs(imgs_camera)
        
        # NOTE: for debugging
        # Image.fromarray(np.transpose(imgs_camera[0,0], (1, 2, 0)).astype('uint8'), 'RGB').show()

        imgs_camera_shape = imgs_camera.shape
        imgs_camera = imgs_camera.reshape(imgs_camera_shape[0]*imgs_camera_shape[1], *imgs_camera_shape[2:])
        imgs_camera = torch.from_numpy(imgs_camera).float().to(self.env.env.device)
        with torch.no_grad():
            emb_camera = self.env.env.embedding(imgs_camera)
            emb_camera = emb_camera.view(imgs_camera_shape[0]*imgs_camera_shape[1], self.env.env.embedding_dim)
        assert self.env.env.proprio == 0, "not implemented for proprio != 0"
        # from embeddings compute the reward with goal
        # convert goal_emb to tensor
        assert len(self.env.env.cameras) == 1, "not implemented for multiple cameras"
        goal_emb = self.env.env.goal_embedding[self.env.env.cameras[0]]
        goal_emb = torch.from_numpy(goal_emb).float().to(self.env.env.device)

        rewards_camera = - torch.norm(emb_camera - goal_emb, dim=-1).to('cpu').numpy()
        rewards_camera = rewards_camera.reshape(imgs_camera_shape[0], imgs_camera_shape[1])
        
        diff_sum = 0
        for i in range(len(paths)):
            paths[i]["post_rewards"] = [rewards_camera[i][-1]]
            # diff_sum += np.linalg.norm(paths[i]["rewards"] - rewards_camera[i], axis=0)
        print(f'average diff rewards diff of 32: {diff_sum / len(paths)}')

        return paths

