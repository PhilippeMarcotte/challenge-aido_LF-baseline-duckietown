#!/usr/bin/env python
import math
import time
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading
import time
import numpy as np
import os
from ddpg import DDPG

class rl_model(object):

    def __init__(self):
        self.node_name = rospy.get_name()
        self.sub_car_cmd = rospy.Subscriber("~car_cmd", Twist2DStamped, self.correct_car_cmd)
        self.sub_img = rospy.Subscriber("~corrected_image/compressed", CompressedImage, self.store_image)
        self.pub_cor_car_cmd = rospy.Publisher("~corrected_car_cmd", Twist2DStamped, queue_size=1)
        rospy.on_shutdown(self.custom_shutdown)

        self.policy = DDPG((480, 640, 3), 2, 1.0)

        self.policy.load("DDPG_123", "/duckietown/catkin_ws/src/rl_model/models")


        rospy.loginfo("[%s] Initialized " % (rospy.get_name()))

    def custom_shutdown(self):
        rospy.loginfo("[%s] Shutting down..." % self.node_name)

        self.sub_car_cmd.unregister()

        # Send stop command
        car_control_msg = Twist2DStamped()
        car_control_msg.v = 0.0
        car_control_msg.omega = 0.0
        self.publishCmd(car_control_msg)

        rospy.sleep(0.5)    #To make sure that it gets published.
        rospy.loginfo("[%s] Shutdown" %self.node_name)
    
    def store_image(self, image_msg):
        self.obs = jpg2rgb(image_msg.data)

    def publishCmd(self, car_cmd_msg):
        self.pub_cor_car_cmd.publish(car_cmd_msg)

    def correct_car_cmd(self, car_cmd):
        correction_msg = Twist2DStamped()
        correction_msg.header = car_cmd.header

        correction = self.policy.predict(self.obs)
        
        correction = 0.15 * correction
        correction_msg.v = car_cmd.v + correction[0]
        correction_msg.omega = car_cmd.omega + correction[1]
        rospy.loginfo([car_cmd.v, car_cmd.omega, correction[0], correction[1], correction_msg.v, correction_msg.omega])
        self.publishCmd(correction_msg)


def jpg2rgb(image_data):
    """ Reads JPG bytes as RGB"""
    from PIL import Image
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
