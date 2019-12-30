#!/usr/bin/env python
from duckietown_msgs.msg import Twist2DStamped, BoolStamped, VehiclePose
import time
import rospy
from collections import deque
import numpy as np

class VehicleAvoidanceControlNode(object):

	def __init__(self):
		self.node_name = rospy.get_name()

		self.desired_distance = self.setupParam('~desired_distance', 0.20)
		self.minimal_distance = self.setupParam('~minimal_distance', 0.35)
		self.Kp = self.setupParam('~Kp', 0.7)
		self.Ki = self.setupParam('~Ki', 0.0)
		self.Kd = self.setupParam('~Kd', 0.0)
		self.Kp_delta_v = self.setupParam('~Kp_delta_v', 0.8)

		self.controllerInitialization()
		self.detection_prev=None
		
		self.car_cmd_pub = rospy.Publisher("~car_cmd",
				Twist2DStamped, queue_size = 1)
		self.vehicle_detected_pub = rospy.Publisher("~vehicle_detected",
				BoolStamped, queue_size=1)
		self.subscriber = rospy.Subscriber("~detection",
				BoolStamped, self.callback,  queue_size=1)
		self.sub_vehicle_pose = rospy.Subscriber("~vehicle_pose", VehiclePose, self.cbPose, queue_size=1)
		self.sub_car_cmd = rospy.Subscriber("~car_cmd_in", Twist2DStamped, self.cbCarCmd, queue_size=1)
		self.init_time = time.time()
		self.last_dist = time.time()
		self.robot_is_in_front = deque(maxlen=20)


	def setupParam(self, param_name, default_value):
		value = rospy.get_param(param_name, default_value)
		rospy.set_param(param_name, value)
		rospy.loginfo("[%s] %s = %s " %(self.node_name,param_name,value))
		return value

	def controllerInitialization(self):
		self.vehicle_pose_msg_temp = VehiclePose()
		self.vehicle_pose_msg_temp.header.stamp = rospy.Time.now()
		self.time_temp = rospy.Time.now()
		self.v_rel = 0
		self.v = 0
		self.detection = False
		self.v_error_temp = 0
		self.I = 0
		self.v_follower = 0
		#self.rho_temp = 0
		self.omega = 0

	def callback(self, data):

		vehicle_detected_msg_out = BoolStamped()
		vehicle_detected_msg_out.header.stamp = data.header.stamp
		vehicle_detected_msg_out.data = data.data
		self.vehicle_detected_pub.publish(vehicle_detected_msg_out)
		self.detection_prev=self.detection
		self.detection = data.data
		self.robot_is_in_front.append(self.detection)


	def cbPose(self, vehicle_pose_msg):
		print("rho: {}".format(vehicle_pose_msg.rho.data))
		print("last_dist: {}".format(time.time()-self.last_dist))
		if vehicle_pose_msg.rho.data < 1:
			self.v = (vehicle_pose_msg.rho.data > self.minimal_distance) * vehicle_pose_msg.rho.data / 5
		else:
			self.v = 1
		self.last_dist = time.time()


	def cbCarCmd(self, car_cmd_msg):
		car_cmd_msg_current = Twist2DStamped()
		car_cmd_msg_current = car_cmd_msg
		car_cmd_msg_current.header.stamp = rospy.Time.now()
		if np.sum(self.robot_is_in_front) > 1:
			car_cmd_msg_current.v *= self.v
			car_cmd_msg_current.omega *= self.v

		if self.detection_prev and not self.detection:
		 	car_cmd_msg_current.v = 0
			car_cmd_msg_current.omega = 0

		print(car_cmd_msg_current.v, car_cmd_msg_current.omega)

		self.car_cmd_pub.publish(car_cmd_msg_current)

if __name__ == '__main__':
	rospy.init_node('vehicle_avoidance_control_node', anonymous=False)
	controller = VehicleAvoidanceControlNode()
	rospy.spin()