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


class ROSAgent(object):
    def __init__(self):
        # Get the vehicle name, which comes in as HOSTNAME
        self.vehicle = os.getenv('HOSTNAME')

        self.ik_action_sub = rospy.Subscriber('/{}/lane_filter_node/lane_pose'.format(
            self.vehicle), LanePose, self._ik_action_cb)
            # rospy.Subscriber("~lane_pose", LanePose, self.error_reader, queue_size=1)
        # Place holder for the action, which will be read by the agent in solution.py
        self.action = np.array([0.0, 0.0])
        self.updated = True

        # Publishes onto the corrected image topic
        # since image out of simulator is currently rectified
        self.cam_pub = rospy.Publisher('/{}/camera_node/image/compressed'.format(
            self.vehicle), CompressedImage, queue_size=10)

        # Publisher for camera info - needed for the ground_projection
        self.cam_info_pub = rospy.Publisher('/{}/camera_info_topic'.format(
            self.vehicle), CameraInfo, queue_size=1)

        # Get args for training
        self.args = get_ddpg_args_train()
        self.dist = None
        self.angle = None
        self.prev_dist = None
        self.prev_angle = None
        
        # Initializes the node
        rospy.init_node('GymDuckietown')

    def init_policy(self, state_dim, action_dim, max_action):
        self.policy = DDPG(state_dim, action_dim, max_action)

    def _ik_action_cb(self, msg):
        """
        Callback to listen to last outputted action from inverse_kinematics node
        Stores it and sustains same action until new message published on topic
        """
        self.prev_dist=self.dist
        self.prev_angle=self.angle

        self.dist = msg.d
        self.angle = msg.phi

        # Select action randomly or according to policy
        if self.total_timesteps < self.args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = self.policy.predict(np.array(obs), np.array(dist), np.array(angle),
            only_pid=total_timesteps-self.args.start_timesteps<self.args.pid_timesteps)
            if self.args.expl_noise != 0:
                action = (action + np.random.normal(
                    0,
                    args.expl_noise,
                    size=env.action_space.shape[0])
                        ).clip(env.action_space.low, env.action_space.high)
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
        img_msg.data = np.array(cv2.imencode('.jpg', contig)[1]).tostring()

        self.obs=contig

        self.cam_pub.publish(img_msg)
        self._publish_info()
        self.callback_processed = False

if __name__ == '__main__':
    rosagent = ROSAgent()
    env = launch_env()
    # env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    env = SteeringToWheelVelWrapper(env)
    env = wrappers.Monitor(env, './gym_results', video_callable=lambda episode_id: True, force=True)
    ################################################################################
    policy_name = "DDPG"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_ddpg_args_train()

    file_name = "{}_{}".format(
        policy_name,
        str(args.seed),
    )

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    logging.getLogger('gym-duckietown').setLevel(logging.ERROR)

    # Set seeds
    seed(args.seed)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    rosagent.init_policy(state_dim, action_dim, max_action)
    # policy = DDPG(state_dim, action_dim, max_action)

    replay_buffer = ReplayBuffer(args.replay_buffer_max_size)

    # Evaluate untrained policy
    # evaluations= [evaluate_policy(env, policy, device)]


    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    episode_reward = None
    env_counter = 0
    while total_timesteps < args.max_timesteps:
        if done:

            if total_timesteps != 0:
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                    total_timesteps, episode_num, episode_timesteps, episode_reward))
                rosagent.policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluations.append(evaluate_policy(env, rosagent.policy, device))

                if args.save_models:
                    rosagent.policy.save(file_name, directory="./pytorch_models")
                np.savez("./results/{}.npz".format(file_name),evaluations)

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
        new_obs, reward, done, _ = env.step(action)
        rosagent.publish_img(new_obs)
        

        time_1=time.time()
        while not rosagent.callback_processed:
            time.sleep(0.001)
        print(time.time()-time_1)

        next_dist = rosagent.dist       # Distance to lane center. Left is negative, right is positive.
        next_angle = rosagent.angle  # Angle from straight, in radians. Left is negative, right is positive.

        if episode_timesteps >= args.env_timesteps:
            done = True

        done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        if first_loop
        replay_buffer.add(obs, new_obs, action, reward, done_bool,
         rosagent.prev_dist, rosagent.prev_angle, next_dist, next_angle)

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        rosagent.total_timesteps = total_timesteps
        timesteps_since_eval += 1

    # Final evaluation
    evaluations.append(evaluate_policy(env, rosagent.policy, device))


    if args.save_models:
        rosagent.policy.save(file_name, directory="./pytorch_models")
    np.savez("./pytorch_models/{}.npz".format(file_name),evaluations)
    ################################################################################

    while not rospy.is_shutdown():
        action = rosagent.action
        obs, reward, done, _ = env.step(action)

        if done:
            obs = env.reset()

        rosagent.publish_img(obs)
        r.sleep()

