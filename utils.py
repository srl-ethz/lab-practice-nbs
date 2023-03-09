import numpy as np
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

def simulate_and_record_video(env, length=100):
    """
    Record a video of the environment for a given number of steps and save it to a file.
    """
    vr = VideoRecorder(env, base_path="video", enabled=True)
    obs = env.reset()
    for _ in range(length):
        action = np.ones(2)
        vr.capture_frame()
        obs, _, _, _ = env.step(action)
    vr.close()
