#!/usr/bin/env python3

import random
import time

import torch
import gym
from gym import wrappers
import os
from duckietown_rl.ddpg import DDPG
from duckietown_rl.args import get_ddpg_args_train
from duckietown_rl.utils import seed, evaluate_policy, ReplayBuffer 
from duckietown_rl.wrappers import DtRewardWrapper, ActionWrapper, SteeringToWheelVelWrapper
from duckietown_rl.object_wrappers import imgWrapper

from collections import deque

# from env import launch_env
import logging
#######################################################################################
import sys
if '/duckietown/' not in sys.path:
    sys.path.append('/duckietown/')

from utils.ros_helpers import launch_env

import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo
from gymdt.msg import Twist2DStamped, WheelsCmdStamped, LanePose
import numpy as np
import os
import cv2
from tensorboardX import SummaryWriter

def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class ROSAgent(object):
    def __init__(self, args, action_space, writer):
        # Get the vehicle name, which comes in as HOSTNAME
        self.vehicle = os.getenv('HOSTNAME')

        self.ik_action_sub = rospy.Subscriber('/{}/lane_controller_node/car_cmd'.format(
            self.vehicle), Twist2DStamped, self._ik_action_cb)
            # rospy.Subscriber("~lane_pose", LanePose, self.error_reader, queue_size=1)
        # Place holder for the action, which will be read by the agent in solution.py
        self.action = None
        self.updated = True

        # Publishes onto the corrected image topic
        # since image out of simulator is currently rectified
        self.cam_pub = rospy.Publisher('/{}/camera_node/image/compressed'.format(
            self.vehicle), CompressedImage, queue_size=10)

        # Publisher for camera info - needed for the ground_projection
        self.cam_info_pub = rospy.Publisher('/{}/camera_info_topic'.format(
            self.vehicle), CameraInfo, queue_size=1)

        # Get args for training
        self.controller_action = None
        self.rl_action = None
        self.total_timesteps = 0

        self.args = args
        self.action_space = action_space

        if not os.path.exists("/duckietown/catkin_ws/results/tensorboard"):
            os.makedirs("/duckietown/catkin_ws/results/tensorboard")
        self.writer = writer
        
        # Initializes the node
        rospy.init_node('GymDuckietown', log_level=rospy.INFO)
        self.evaluation = False

    def init_policy(self, state_dim, action_dim, max_action):
        self.policy = DDPG(state_dim, action_dim, max_action)

    def _ik_action_cb(self, msg):
        """
        Callback to listen to last outputted action from inverse_kinematics node
        Stores it and sustains same action until new message published on topic
        """
        self.controller_action = np.array([msg.v, msg.omega])
        # Select action randomly or according to policy
        if self.total_timesteps < self.args.start_timesteps and not self.evaluation:
            rospy.logerr_once("RANDOM")
            self.rl_action = self.action_space.sample()
            action = self.rl_action
        elif self.total_timesteps < self.args.controller_timesteps and not self.evaluation:
            rospy.logerr_once("CONTROLLER")
            action = self.controller_action
        else:
            rospy.logerr_once("RL")
            self.rl_action = self.policy.predict(np.array(self.obs))
            action = self.controller_action + self.rl_action

            if self.args.expl_noise != 0:
                action = (action + np.random.normal(
                    0,
                    self.args.expl_noise,
                    size=self.action_space.shape[0])
                        ).clip(self.action_space.low, self.action_space.high)
        self.action = action
        self.updated = True
        self.callback_processed = True

    def _publish_info(self):
        """
        Publishes a default CameraInfo
        """

        self.cam_info_pub.publish(CameraInfo())

    def publish_img(self, obs, evaluation=False):
        """
        Publishes the image to the compressed_image topic, which triggers the lane following loop
        """
        self.evaluation = evaluation
        img_msg = CompressedImage()

        time = rospy.get_rostime()
        img_msg.header.stamp.secs = time.secs
        img_msg.header.stamp.nsecs = time.nsecs

        img_msg.format = "jpeg"
        contig = cv2.cvtColor(np.ascontiguousarray(obs), cv2.COLOR_BGR2RGB)
        data = np.array(cv2.imencode('.jpg', contig)[1])
        img_msg.data = data.tostring()

        self.obs = obs

        self.cam_pub.publish(img_msg)
        self._publish_info()
        self.callback_processed = False

if __name__ == '__main__':
    args = get_ddpg_args_train()
    policy_name = "DDPG"
    file_name = "{}_{}".format(
        policy_name,
        str(args.seed),
    )
    writer = SummaryWriter("/duckietown/catkin_ws/results/tensorboard/{}".format(file_name))

    seed(args.seed)

    env = launch_env()
    # env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    env = SteeringToWheelVelWrapper(env)
    rosagent = ROSAgent(args, env.action_space, writer)

    # env = wrappers.Monitor(env, '/duckietown/gym_results', video_callable=lambda episode_id: True, force=True)
    ################################################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists("/duckietown/catkin_ws/results"):
        os.makedirs("/duckietown/catkin_ws/results")
    if args.save_models and not os.path.exists("/duckietown/catkin_ws/pytorch_models"):
        os.makedirs("/duckietown/catkin_ws/pytorch_models")

    logging.getLogger('gym-duckietown').setLevel(logging.ERROR)
    logging.getLogger("rosout").setLevel(logging.INFO)

    # Set seeds
    seed(args.seed)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    rosagent.init_policy(state_dim, action_dim, max_action)
    # policy = DDPG(state_dim, action_dim, max_action)

    replay_buffer = ReplayBuffer(args.replay_buffer_max_size)


    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    episode_reward = None
    env_counter = 0
    obs = env.reset()

    rospy.logerr("WAITING LINE DETECTION")
    while rosagent.action is None:
        rosagent.publish_img(obs)

    rospy.logerr("STARTING")

    # Evaluate untrained policy
    evaluations= [evaluate_policy(env, rosagent, device)]
    writer.add_scalar("eval.reward", evaluations[-1], total_timesteps)
    time_avg = deque(maxlen=10000)
    while total_timesteps < args.max_timesteps:
        start = time.time()
        if done:

            if total_timesteps != 0:
                rospy.logerr(("Total T: %d Episode Num: %d Episode T: %d Reward: %f Duration: %.2f s Time Left: %.2f h") % (
                    total_timesteps, episode_num, episode_timesteps, episode_reward, 
                    np.average(time_avg), np.average(time_avg) * (args.max_timesteps - total_timesteps) / 3600.0))
                rosagent.policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, 
                 only_critic=False)

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluations.append(evaluate_policy(env, rosagent, device))
                writer.add_scalar("eval.reward", evaluations[-1], total_timesteps)
                rospy.logerr("Policy evaluation: %f" % (evaluations[-1]))

                if args.save_models:
                    rosagent.policy.save(file_name, directory="/duckietown/catkin_ws/pytorch_models")
                np.savez("/duckietown/catkin_ws/results/{}.npz".format(file_name),evaluations)

            # Reset environment
            env_counter += 1
            # env.close()
            obs = env.reset()
            rosagent.publish_img(obs)
            while not rosagent.callback_processed:
                continue
            # env.reset_video_recorder()
            # env = wrappers.Monitor(env, './gym_results', video_callable=lambda episode_id: True, force=True)
            if episode_reward is not None:
                writer.add_scalar("train.reward", episode_reward, total_timesteps)
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        action=rosagent.action

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        rosagent.publish_img(new_obs)
        while not rosagent.callback_processed:
            continue

        writer.add_scalar("train.controller.action.v", np.abs(rosagent.controller_action[0]), total_timesteps)
        writer.add_scalar("train.controller.action.omega", np.abs(rosagent.controller_action[1]), total_timesteps)
        writer.add_scalar("train.rl.action.v", np.abs(rosagent.rl_action[0]), total_timesteps)
        writer.add_scalar("train.rl.action.omega", np.abs(rosagent.rl_action[1]), total_timesteps)

        if episode_timesteps >= args.env_timesteps:
            done = True

        done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add(imgWrapper(obs), imgWrapper(new_obs), rosagent.controller_action, rosagent.rl_action, reward, done_bool)

        episode_timesteps += 1
        total_timesteps += 1
        rosagent.total_timesteps = total_timesteps
        timesteps_since_eval += 1

        obs = new_obs
        time_avg.append(time.time() - start)

    # Final evaluation
    evaluations.append(evaluate_policy(env, rosagent, device))


    if args.save_models:
        rosagent.policy.save(file_name, directory="/duckietown/catkin_ws/pytorch_models")
    np.savez("/duckietown/catkin_ws/pytorch_models/{}.npz".format(file_name),evaluations)
    ################################################################################
