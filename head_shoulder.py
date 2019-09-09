import sys
import rospy
from std_msgs.msg import String
#from trajectory_msgs.msg import JointTrajectory
from opencv_apps.msg import RotatedRectStamped
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from naoqi_bridge_msgs.msg import HeadTouch

message_arrived=False
head_touch_message_arrived=False
def talker6(msg):
    global message_arrived
    message_arrived=msg
def headtouch_cb(msg):
    global head_touch_message_arrived
    head_touch_message_arrived=msg

if __name__=='__main__':
    try:
        message_arrived=False
        rospy.init_node('talker6')
        rospy.Subscriber('/camshift/track_box',RotatedRectStamped,talker6)
        rospy.Subscriber('/pepper_robot/head_touch',HeadTouch,headtouch_cb)
        pub=rospy.Publisher('/pepper_robot/pose/joint_angles',JointAnglesWithSpeed,queue_size=1)
        #pub=rospy.Publisher('/pepper_robot/pose/joint_angles',JointAnglesWithSpeed,queue_size=1)
        rospy.sleep(1) #omajinai
        while not rospy.is_shutdown():
            hello=JointAnglesWithSpeed();
            #if message_arrived:
            if  head_touch_message_arrived and head_touch_message_arrived.button==1 and head_touch_message_arrived.state==1:
                print("message_arrived")
                if True:#(message_arrived.rect.size.width*message_arrived.rect.size.height>4000):
                    hello.joint_names.append("LShoulderPitch")
                    hello.joint_names.append("RShoulderPitch")

                    hello.speed=0.05
                    #hello.relative=0
                    hello.joint_angles=[-0.05,-0.05]

                    print(message_arrived)

            message_arrived=False
            pub.publish(hello);

            rospy.sleep(1)
            #sys.exit(0)

    except rospy.ROSInterruptException:
        pass
