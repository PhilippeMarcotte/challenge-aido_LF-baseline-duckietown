import gym
import gym_duckietown

def launch_env(id=None):
    env = None
    if id is None:
        from gym_duckietown.simulator import Simulator
        env = Simulator(
            seed=123, # random seed
            map_name="loop_empty",
            # it is important to set to the max length of one episode for the video recording
            max_steps=500, # we don't want the gym to reset itself
            domain_rand=0,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4, # start close to straight
            full_transparency=True,
            distortion=True,
        )
    else:
        env = gym.make(id)

    return env
