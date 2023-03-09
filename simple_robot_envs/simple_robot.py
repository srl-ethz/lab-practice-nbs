import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import mujoco
import mujoco_viewer
import os

class SimpleTorqueControlledRobot(gym.Env):
    """
    Simple gym implementation of a torque-controlled 2-joint robot, simulated in mujoco
    rendering has not been tested yet
    https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, render_mode=None):

        # set up mujoco simulation
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model = mujoco.MjModel.from_xml_path(os.path.join(current_dir, "simple_robot.xml"))
        self.data = mujoco.MjData(self.model)
        self.num_joints = self.model.njnt

        # set up gym parameters
        self.action_space = Box(low=-1, high=1, shape=(self.num_joints,))
        # joint range is -90 to 90 degrees, but mujoco's soft constraints may go beyond that,
        # so set a larger range here to leave some slack
        self.observation_space = Box(low=-np.pi, high=np.pi, shape=(self.num_joints*2,))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "rgb_array":
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, "offscreen")
        elif self.render_mode == "human":
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
    
    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        return np.concatenate([qpos, qvel])

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = 0
        done = False

        if self.render_mode == "human":
            self.viewer.render()

        return obs, reward, done, {}
    
    def close(self):
        pass

    def reset(self, seed=None):
        # seed the random number generator
        super().reset(seed=seed)
        # TODO: set random initial state
        mujoco.mj_resetData(self.model, self.data)
        obs = self._get_obs()
        return obs
    
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        img = self.viewer.read_pixels()
        return img