import sys
import time
import math
from naoqi import ALProxy


def main(robotIP):
	PORT = 9559


	# 1. Create proxies
	try:
    	motionProxy  = ALProxy("ALMotion", robotIP, PORT)
    	postureProxy = ALProxy("ALRobotPosture", robotIP, PORT)
    	sonarProxy   = ALProxy("ALSonar", robotIP, PORT)
    	memoryProxy  = ALProxy("ALMemory", robotIP, PORT)
	except Exception as e:
    	print "Could not create proxy to ALModule"
    	print "Error was: ", e
    	sys.exit(1)


	# 2. Setup
	print "Waking up..."
	motionProxy.wakeUp()
	postureProxy.goToPosture("StandInit", 0.5)
    
	# Subscribe to sonars
	sonarProxy.subscribe("BlindWalker")


	l_sonar_key = "Device/SubDeviceList/US/Left/Sensor/Value"
	r_sonar_key = "Device/SubDeviceList/US/Right/Sensor/Value"


	print "Starting autonomous wandering. Press Ctrl+C to stop."
    
	# --- CONFIGURATION ---
	walking_speed = 0.5 	# Forward speed
	obs_threshold = 0.2 	# UPDATED: 20 cm
	turn_angle = 120    	# UPDATED: Degrees to turn
	# Convert degrees to radians for the robot
	turn_radians = turn_angle * (math.pi / 180.0)
	# ---------------------


	try:
    	while True:
        	# Get sonar values
        	val_left = memoryProxy.getData(l_sonar_key)
        	val_right = memoryProxy.getData(r_sonar_key)


        	# Check if EITHER sensor is too close
        	if (val_left < obs_threshold) or (val_right < obs_threshold):
            	print "Obstacle Detected! (L: %.2f, R: %.2f)" % (val_left, val_right)
           	 
            	# 1. STOP immediately
            	motionProxy.stopMove()
           	 
            	# 2. DECIDE direction
            	# If left is closer, turn right (negative angle)
            	# If right is closer, turn left (positive angle)
            	if val_left < val_right:
                	print "Turning Right 120 degrees..."
                	target_turn = -turn_radians
            	else:
                	print "Turning Left 120 degrees..."
                	target_turn = turn_radians
           	 
            	# 3. EXECUTE PRECISE TURN
            	# moveTo(x, y, theta) is a blocking call.
            	# The script will pause here until the turn is complete.
            	# x=0, y=0 ensures it turns in place.
            	motionProxy.moveTo(0.0, 0.0, target_turn)
           	 
        	else:
            	# Path is clear, keep walking
            	motionProxy.move(walking_speed, 0.0, 0.0)
       	 
        	# Loop delay
        	time.sleep(0.2)


	except KeyboardInterrupt:
    	print "Interrupted by user, shutting down."
   	 
	finally:
    	print "Stopping..."
    	motionProxy.stopMove()
    	sonarProxy.unsubscribe("BlindWalker")
    	motionProxy.rest()


if __name__ == "__main__":
	if len(sys.argv) <= 1:
    	print "Usage: python wander_nao.py <ROBOT_IP>"
	else:
    	main(sys.argv[1])





