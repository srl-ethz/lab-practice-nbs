import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import mujoco
import os

# Set up fake display; otherwise rendering will fail
import os
os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'

class SimpleTorqueControlledRobot(gym.Env):
    """
    Simple gym implementation of a torque-controlled 2-joint robot, simulated in mujoco
    references:
    https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py
    https://pypi.org/project/mujoco/ (the Colab notebook linked here is a good sample for using mujoco in python)
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    def __init__(self, render_mode="rgb_array"):

        # set up mujoco simulation
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model = mujoco.MjModel.from_xml_path(os.path.join(current_dir, "simple_robot.xml"))
        self.data = mujoco.MjData(self.model)
        self.num_joints = self.model.njnt

        # set up gym parameters
        self.action_space = Box(low=-1, high=1, shape=(self.num_joints,))
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_joints*2,))

        # set up rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "rgb_array":
            # self.ctx = mujoco.GLContext(600, 600)
            self.renderer = mujoco.Renderer(self.model)
    
    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        return np.concatenate([qpos, qvel], dtype=np.float32)

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = 0
        done = False

        return obs, reward, done, {}
    
    def close(self):
        pass

    def reset(self, seed=None, options=None):
        # seed the random number generator
        super().reset(seed=seed)
        # TODO: set random initial state
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)  # ensure rendering can be done
        obs = self._get_obs()
        return obs, {}

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        self.renderer.update_scene(self.data)
        img = self.renderer.render()
        return img.copy()