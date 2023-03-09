from gymnasium.envs.registration import register

# register custom simple robot environments to be detected by gym

register(
    id='SimpleTorqueControlledRobot',
    entry_point='simple_robot_envs.simple_robot:SimpleTorqueControlledRobot',
)