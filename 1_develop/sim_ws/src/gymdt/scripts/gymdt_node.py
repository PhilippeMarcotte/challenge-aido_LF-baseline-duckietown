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

def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class ROSAgent(object):
    def __init__(self, args, action_space):
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
        self.v = None
        self.omega = None
        self.prev_v = None
        self.prev_omega = None
        self.total_timesteps = 0

        self.args = args
        self.action_space = action_space
        
        # Initializes the node
        rospy.init_node('GymDuckietown', log_level=rospy.INFO)

    def init_policy(self, state_dim, action_dim, max_action):
        self.policy = DDPG(state_dim, action_dim, max_action)

    def _ik_action_cb(self, msg):
        """
        Callback to listen to last outputted action from inverse_kinematics node
        Stores it and sustains same action until new message published on topic
        """
        self.prev_v=self.v
        self.prev_omega=self.omega

        self.v = msg.v
        self.omega = msg.omega
        # Select action randomly or according to policy
        if self.total_timesteps < self.args.start_timesteps:
            rospy.logerr_once("RANDOM")
            action = self.action_space.sample()
        elif self.total_timesteps < self.args.controller_timesteps:
            rospy.logerr_once("CONTROLLER")
            action = np.array([msg.v, msg.omega])
        else:
            rospy.logerr_once("RL")
            action = self.policy.predict(np.array(self.obs), np.array(self.v), np.array(self.omega))
            if self.args.expl_noise != 0:
                action = (action + np.random.normal(
                    0,
                    self.args.expl_noise,
                    size=self.action_space.shape[0])
                        ).clip(self.action_space.low, self.action_space.high)
        self.prev_action = self.action
        self.action = action
        self.updated = True
        self.callback_processed = True

    def _publish_info(self):
        """
        Publishes a default CameraInfo
        """

        self.cam_info_pub.publish(CameraInfo())

    def publish_img(self, obs):
        """
        Publishes the image to the compressed_image topic, which triggers the lane following loop
        """
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

    seed(args.seed)

    env = launch_env()
    # env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    env = SteeringToWheelVelWrapper(env)
    rosagent = ROSAgent(args, env.action_space)

    # env = wrappers.Monitor(env, '/duckietown/gym_results', video_callable=lambda episode_id: True, force=True)
    ################################################################################
    policy_name = "DDPG"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_ddpg_args_train()

    file_name = "{}_{}".format(
        policy_name,
        str(args.seed),
    )

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
    evaluations= [evaluate_policy(env, rosagent.policy, device)]
    while total_timesteps < args.max_timesteps:
        start = time.time()
        if done:

            if total_timesteps != 0:
                rospy.logerr(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                    total_timesteps, episode_num, episode_timesteps, episode_reward))
                rosagent.policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, 
                 only_critic=total_timesteps < args.controller_timesteps)

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluations.append(evaluate_policy(env, rosagent.policy, device))
                rospy.logerr("Policy evaluation: %f" % (evaluations[-1]))

                if args.save_models:
                    rosagent.policy.save(file_name, directory="/duckietown/catkin_ws/pytorch_models")
                np.savez("/duckietown/catkin_ws/results/{}.npz".format(file_name),evaluations)

            # Reset environment
            env_counter += 1
            # env.close()
            obs = env.reset()
            rosagent.publish_img(obs)
            # env.reset_video_recorder()
            # env = wrappers.Monitor(env, './gym_results', video_callable=lambda episode_id: True, force=True)
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        action=rosagent.action

        # Perform action
        new_obs, reward, done, _ = env.step(action if action is not None else np.array([0.0, 0.0]))
        rosagent.publish_img(new_obs)

        #while not rosagent.callback_processed:
        #    time.sleep(0.001)
        #print(time.time()-time_1)

        if episode_timesteps >= args.env_timesteps:
            done = True

        done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
        episode_reward += reward

        # Store data in replay buffer

        if rosagent.action is not None:
            replay_buffer.add(imgWrapper(obs), imgWrapper(new_obs), action, reward, done_bool,
            rosagent.prev_v, rosagent.prev_omega, rosagent.v, rosagent.omega)

            episode_timesteps += 1
            total_timesteps += 1
            rosagent.total_timesteps = total_timesteps
            timesteps_since_eval += 1

        obs = new_obs

    # Final evaluation
    evaluations.append(evaluate_policy(env, rosagent.policy, device))


    if args.save_models:
        rosagent.policy.save(file_name, directory="/duckietown/catkin_ws/pytorch_models")
    np.savez("/duckietown/catkin_ws/pytorch_models/{}.npz".format(file_name),evaluations)
    ################################################################################
