import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import mujoco

class SimpleTorqueControlledRobot(gym.Env):
    """
    haven't checked if this works yet
    """
    def __init__(self):
        self.num_joints = 2

        # set up gym parameters
        self.action_space = Box(low=-1, high=1, shape=(self.num_joints,))
        # joint range is -90 to 90 degrees, but mujoco's soft constraints may go beyond that,
        # so set a larger range here to leave some slack
        self.observation_space = Box(low=-np.pi, high=np.pi, shape=(self.num_joints*2,))
        self.model = mujoco.load_model_from_path('simple_robot.xml')
        self.sim = mujoco.MjSim(self.model)

    def step(self, action):
        self.sim.data.ctrl[:] = action
        self.sim.step()
        obs = self.sim.data.qpos.flat.copy()
        obs = np.concatenate([obs, self.sim.data.qvel.flat.copy()])
        reward = -np.sum(np.square(obs))
        done = False
        return obs, reward, done, {}
    
    def close(self):
        pass

    def reset(self):
        self.sim.reset()
        obs = self.sim.data.qpos.flat.copy()
        obs = np.concatenate([obs, self.sim.data.qvel.flat.copy()])
        return obs