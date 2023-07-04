import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

try:
    import mujoco
except ImportError as e:
    MUJOCO_IMPORT_ERROR = e
else:
    MUJOCO_IMPORT_ERROR = None

import math
from typing import Optional, Union
from envs.utils.ma_xml import create_multiagent_xml


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class MultiAntEnv(MujocoEnv, utils.EzPickle):
    """
    ## Description
    This environment is based on the single agent mujoco environment from gymnasium ant_v4.py
    
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        agent_num=1,
        xml_file="ant.xml",
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        args=None,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            ctrl_cost_weight,
            use_contact_forces,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self.args = args

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._use_contact_forces = use_contact_forces

        self.agent_num = agent_num

        #----------------------- single agent observation dim -----------------------#
        # 15: sa_qpos
        # 14: sa_qvel
        # 2 : relative x pos and relative x vel
        # 78: mjtNum* cfrc_ext; // com-based external force on body (nbody x 6). nbody==13. 
        #     In single ant env, nbody==14 that contains 13 bodies and 1 worldbody
        #     In 2 ant env, nbody==27 that contains 2*13 bodies and 1 worldbody
        self.sa_obs_dim = (15 + 14 + 2 * self.agent_num)
        if use_contact_forces:
            self.sa_obs_dim += 78

        # observation space definition
        sa_obs_space = Box(
            low=-np.inf, high=np.inf, shape=(1, self.sa_obs_dim), dtype=np.float64
        )
        ma_obs_space = Box(
            low=-np.inf, high=np.inf, shape=(self.agent_num, self.sa_obs_dim), dtype=np.float64
        )
        observation_space = ma_obs_space

        #----------------------- single agent action dim -----------------------#
        self.sa_action_dim = 8

        # extract agent from base xml
        local_position = "../envs/assets/xml/" + xml_file
        self.fullxml = create_multiagent_xml(local_position, args)

        MujocoEnv.__init__(
            self,
            xml_file,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def step(self, action):
        # input action is a list
        action = np.hstack(action)
        
        ma_xy_position_before = self.ma_position[:, :2]
        self.do_simulation(action, self.frame_skip)
        ma_xy_position_after = self.ma_position[:, :2]
        
        ma_xy_velocity = (ma_xy_position_after - ma_xy_position_before) / self.dt

        ma_forward_reward = self.get_ma_forward_reward(ma_xy_velocity)
        ma_healthy_reward = self.ma_healthy_reward
        
        ma_rewards = ma_forward_reward + ma_healthy_reward

        ma_costs = ma_ctrl_cost = self.get_ma_control_cost(action)

        terminateds = self.terminateds
        observations = self._get_ma_obs()
        info = {    
            "ma_reward_forward": ma_forward_reward,
            "ma_reward_ctrl": -ma_ctrl_cost,
            "ma_reward_survive": ma_healthy_reward,
            "ma_x_position": ma_xy_position_after[:, 0],
            "ma_y_position": ma_xy_position_after[:, 1],
            "ma_distance_from_origin": np.linalg.norm(ma_xy_position_after[:, :2]-self.initial_ma_position[:, :2], ord=2),
            "ma_x_velocity": ma_xy_velocity[:, 0],
            "ma_y_velocity": ma_xy_velocity[:, 1],
            "ma_forward_reward": ma_forward_reward,
        }

        if self._use_contact_forces:
            ma_contact_cost = self.ma_contact_cost
            ma_costs += ma_contact_cost
            info["ma_reward_ctrl"] = -ma_contact_cost

        rewards = [(ma_rewards[idx] - ma_costs[idx]) for idx in range(self.agent_num)]

        # if cfg.COMP.USE_COMPETITION:
        #     # The last one gets 0, and the first one gets N. N equals to agent number.
        #     # competition_reward = ma_xy_position_after[:, 0].argsort().argsort() * cfg.COMP.COMPETITION_REWARD_COEF
        #     competition_reward = ma_xy_velocity[:, 0].argsort().argsort() * cfg.COMP.COMPETITION_REWARD_COEF
        #     rewards = [(rewards[idx] + competition_reward[idx]) for idx in range(self.agent_num)]
        
        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminateds, False, info


    def _initialize_simulation(self):
        # import model from xml string
        self.model = mujoco.MjModel.from_xml_string(self.fullxml)
        # MjrContext will copy model.vis.global_.off* to con.off*
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

    @property
    def ma_healthy_reward(self):
        return (
            np.array([
                (self.ma_is_healthy[idx] or self._terminate_when_unhealthy)
            * self._healthy_reward for idx in range(self.agent_num)
            ])[:, np.newaxis]
        )

    def get_ma_control_cost(self, action):
        # break action into sub actions
        ma_action = self.get_ma_action(action)
        ma_control_cost = np.array([
            self._ctrl_cost_weight * np.sum(np.square(ma_action[idx])) for idx in range(self.agent_num)
            ])[:, np.newaxis]
        return ma_control_cost
    
    def get_ma_action(self, action):
        """ Break action into sub actions corresponding to each agent. """
        assert len(action) % self.agent_num == 0, "Action cannot be aligned!"
        agent_action_length = int(len(action) / self.agent_num)
        ma_action = np.array([action[(idx-1)*agent_action_length:idx*agent_action_length] 
                              for idx in range(1, self.agent_num+1)])
        return ma_action

    @property
    def ma_contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext[1:, :] # first line contact friction data is worldbody?
        assert len(raw_contact_forces) % self.agent_num == 0, "Contact forces cannot be aligned!"
        agent_contact_length = int(len(raw_contact_forces) / self.agent_num)
        ma_contact_forces = np.array([raw_contact_forces[(idx-1)*agent_contact_length:idx*agent_contact_length] 
                              for idx in range(1, self.agent_num+1)])
        
        min_value, max_value = self._contact_force_range
        ma_contact_forces = np.array([
            np.clip(ma_contact_forces[idx], min_value, max_value) for idx in range(self.agent_num)
            ])
        return ma_contact_forces

    @property
    def ma_contact_cost(self):
        ma_contact_cost = np.array([
            self._contact_cost_weight * np.sum(np.square(self.ma_contact_forces[idx])) for idx in range(self.agent_num)
            ])[:, np.newaxis]
        return ma_contact_cost

    @property
    def ma_is_healthy(self):
        min_z, max_z = self._healthy_z_range
        ma_is_healthy = np.array([
            np.isfinite(self.ma_qpos[idx]).all() and 
            np.isfinite(self.ma_qvel[idx]).all() and 
            min_z <= self.ma_qpos[idx][2] <= max_z for idx in range(self.agent_num)]
            )
        return ma_is_healthy

    @property
    def terminateds(self):
        # terminateds = not self.ma_is_healthy.all() if self._terminate_when_unhealthy else False
        # return [terminateds] * self.agent_num
        terminateds = [not is_healthy for is_healthy in self.ma_is_healthy]
        return terminateds

    def _get_ma_obs(self):
        return [self._get_sa_obs(idx) for idx in range(self.agent_num)]


    def _get_sa_obs(self, idx):
        """ Return single agent observation. """
        #--------------- Proprioceptive observations -----------------------#
        sa_qpos = self.ma_qpos[idx]
        sa_qvel = self.ma_qvel[idx]
        sa_contact_force = self.ma_contact_forces[idx].flat.copy()

        #------------------ External observations --------------------------#
        sa_relative_obs = np.hstack(self.get_sa_relative_obs(idx))
        
        #----------------- Concatenate observations ------------------------#
        if self._use_contact_forces:
            sa_obs = np.concatenate((sa_qpos, sa_qvel, sa_contact_force))
        else:
            sa_obs = np.concatenate((sa_qpos, sa_qvel))

        sa_obs = np.concatenate((sa_obs, sa_relative_obs))
        return sa_obs

        
    def get_sa_relative_obs(self, idx):
        """ Return a list including relative observation infos. """
        # single agent position and velocity
        sa_position = self.ma_position[idx]
        sa_velocity = self.ma_velocity[idx]

        sa_relative_obs = []
        for j in range(self.agent_num):
            # neighbours position and velocity
            j_position = self.ma_position[j]
            j_velocity = self.ma_velocity[j]
            # relative position and velocity
            r_position = j_position - sa_position
            r_velocity = j_velocity - sa_velocity

            j_relative_obs = np.concatenate((r_position[:1], r_velocity[:1]))
            if not self.args.use_relative_obs:
                j_relative_obs = np.zeros((2, ))
            if self.args.use_noise:
                j_relative_obs = np.random.random((2, ))
            sa_relative_obs.append(j_relative_obs)
        
        return sa_relative_obs
    
    @property
    def ma_qpos(self):
        """ Return an array that includes the pos of each agent. """
        assert len(self.data.qpos.ravel().copy()) % self.agent_num == 0, "State qpos cannot be aligned!"
        agent_qpos_length = int(len(self.data.qpos.ravel().copy()) / self.agent_num)
        ma_qpos = np.array([self.data.qpos.ravel().copy()[idx*agent_qpos_length:(idx+1)*agent_qpos_length] 
                              for idx in range(self.agent_num)])
        return ma_qpos
    
    @property
    def ma_qvel(self):
        """ Return an array that includes the vel of each agent. """
        assert len(self.data.qvel.ravel().copy()) % self.agent_num == 0, "State qvel cannot be aligned!"
        agent_qvel_length = int(len(self.data.qvel.ravel().copy()) / self.agent_num)
        ma_qvel = np.array([self.data.qvel.ravel().copy()[idx*agent_qvel_length:(idx+1)*agent_qvel_length] 
                              for idx in range(self.agent_num)])
        return ma_qvel

    @property
    def ma_position(self):
        """ Return an array to get multi agent x y z positions. """
        return self.ma_qpos[:, :3]
    
    @property
    def ma_velocity(self):
        """ Return an array to get multi agent x y z velocities. """
        return self.ma_qvel[:, :3]
        
    def get_ma_forward_reward(self, ma_xy_velocity):
        """ Return multi agent forward reward. """
        # 0 denote x, 1 denote y
        ma_forward_reward = np.array([ma_xy_velocity[idx, 0] for idx in range(self.agent_num)])[:, np.newaxis]
        return ma_forward_reward

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        #----------- set initial positions for multi agent ---------------#
        init_ma_qpos = self.ma_qpos
        init_ma_qvel = self.ma_qvel
        for idx in range(self.agent_num):
            if self.args.init_formation == "star":
                r = 1.0 * math.log(self.agent_num)
                init_ma_qpos[idx][0] += r * math.cos(2 * math.pi * idx / self.agent_num)
                init_ma_qpos[idx][1] += r * math.sin(2 * math.pi * idx / self.agent_num)
            elif self.args.init_formation == "line":
                init_ma_qpos[idx][1] += 1.0 * 2 * (idx - (self.agent_num-1) / 2)

        # record agents initial xy position
        self.initial_ma_position = self.ma_position
        # flatten
        init_ma_qpos = init_ma_qpos.ravel()
        init_ma_qvel = init_ma_qvel.ravel()
        # add noise
        init_ma_qpos = init_ma_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        init_ma_qvel = (
            init_ma_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        # set state
        self.set_state(init_ma_qpos, init_ma_qvel)

        observation = self._get_ma_obs()

        return observation
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None,):
        super().reset(seed=seed)

        self._reset_simulation()

        ob = self.reset_model()
        if self.render_mode == "human":
            self.render()
        return ob, {}