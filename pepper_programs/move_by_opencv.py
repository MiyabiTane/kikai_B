import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from opencv_apps.msg import RotatedRectStamped

message_arrived=False
def talker3(msg):
    global message_arrived
    #msg=size,width,height etc..
    message_arrived=msg
    #print(msg)
    """while not rospy.is_shutdown():
        if message_arrived:
            str="hello world %s"%rospy.get_time()
            rospy.loginfo(str)
            hello=Twist();
            hello.linear.x=1;
            hello.linear.y=0;
            hello.linear.z=0;
            hello.angular.x=0;
            hello.angular.y=0;
            hello.angular.z=0;
            pub.publish(hello);
            rospy.sleep(0.2)
        message_arrived=False"""

if __name__=='__main__':
    try:
        message_arrived=False
        rospy.init_node('talker3') #initialize
        rospy.Subscriber('/camshift/track_box',RotatedRectStamped,talker3)
        pub=rospy.Publisher('/cmd_vel',Twist,queue_size=1)
        while not rospy.is_shutdown():
            #print(message_arrived)
            hello=Twist();
            if message_arrived:
                print("message_arrived")
                if (message_arrived.rect.size.width*message_arrived.rect.size.height>4000):
                    #hello=Twist();
                    hello.linear.x=0;
                    hello.linear.y=0;
                    hello.linear.z=0;
                    hello.angular.x=0;
                    hello.angular.y=0;
                    if message_arrived.rect.center.x>160:
                        hello.angular.z=-0.3;
                    else:
                        hello.angular.z=0.3;
                    #pub.publish(hello);
                    #rospy.sleep(0.2)
                    print(message_arrived)

            message_arrived=False
            pub.publish(hello);

            rospy.sleep(0.5)
        #talker3()

    except rospy.ROSInterruptException:
        pass
