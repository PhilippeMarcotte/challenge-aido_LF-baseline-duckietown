#!/usr/bin/env python
import math
import time
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, NumpyArray
from rospy.numpy_msg import numpy_msg
import time
import numpy as np
import os
from ddpg import DDPG
import cv2


class rl_model(object):

    def __init__(self):
        self.node_name = rospy.get_name()
        self.bridge = CvBridge()
        self.policy = DDPG((480, 640, 3), 2, 1.0)
        state_dict = self.policy.actor.state_dict()
        self.policy.load("DDPG_123", "/duckietown/catkin_ws/src/rl_model/models", pickle=False)
        state_dict2 = self.policy.actor.state_dict()
        shared_items = {k: state_dict[k] for k in state_dict if k in state_dict2 and (state_dict[k] == state_dict2[k]).all()}
        rospy.loginfo
        self.policy.actor.eval()
        self.policy.critic.eval()

        self.sub_wheels_cmd = rospy.Subscriber("~wheels_cmd", WheelsCmdStamped, self.correct_wheels_cmd)
        self.sub_img = rospy.Subscriber("~corrected_image/compressed", CompressedImage, self.store_image)
        
        self.pub_corrected_wheels_cmd = rospy.Publisher("~corrected_wheels_cmd", WheelsCmdStamped, queue_size=1)
        self.pub_image = rospy.Publisher("~image", Image, queue_size=1)

        rospy.on_shutdown(self.custom_shutdown)
        self.count = 0


        rospy.loginfo("[%s] Initialized " % (rospy.get_name()))

    def custom_shutdown(self):
        rospy.loginfo("[%s] Shutting down..." % self.node_name)

        self.sub_wheels_cmd.unregister()

        # Send stop command
        wheels_control_msg = WheelsCmdStamped()
        wheels_control_msg.vel_left = 0.0
        wheels_control_msg.vel_right = 0.0
        self.publish_cmd(wheels_control_msg)

        rospy.sleep(0.5)    #To make sure that it gets published.
        rospy.loginfo("[%s] Shutdown" %self.node_name)
    
    def store_image(self, image_msg):
        self.obs = jpg2rgb(image_msg.data)

    def publish_cmd(self, wheels_cmd_msg):
        self.pub_corrected_wheels_cmd.publish(wheels_cmd_msg)

    def correct_wheels_cmd(self, wheels_cmd):
        correction_msg = WheelsCmdStamped()
        correction_msg.header = wheels_cmd.header

        correction = self.policy.predict(np.array(self.obs))

        noise = np.random.normal(
            0,
            0.1,
            size=2)
        # correction += noise.clip(-1, 1)
        
        # correction = 0.5 * (correction + 1)
        correction_msg.vel_left = wheels_cmd.vel_left + correction[0]
        correction_msg.vel_right = wheels_cmd.vel_right + correction[1]
        # rospy.loginfo("Wheels cmd: {}".format([wheels_cmd.vel_left, wheels_cmd.vel_right]))
        # rospy.loginfo("Correction: {}".format([correction[0], correction[1]]))
        # rospy.loginfo("Total: {}".format([correction_msg.vel_left, correction_msg.vel_right]))
        self.publish_cmd(correction_msg)


def jpg2rgb(image_data):
    """ Reads JPG bytes as RGB"""
    from PIL import Image
    import cv2
    import io
    im = Image.open(io.BytesIO(image_data))
    im = im.convert('RGB')
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data


if __name__ == "__main__":
    rospy.init_node("rl_model_node", anonymous=False)  # adapted to sonjas default file

    rl_model_node = rl_model()
    rospy.spin()
