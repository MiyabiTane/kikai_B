import rospy
from std_msgs.msg import String
from naoqi_bridge_msgs.msg import HeadTouch

#msg=size etc..
head_message_arrived=False
def talker56(msg):
    global head_message_arrived
    head_message_arrived=msg
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
        head_message_arrived=False
        rospy.init_node('talker56')
        rospy.Subscriber('/pepper_robot/head_touch',HeadTouch,talker56)
        pub=rospy.Publisher('/speech',String,queue_size=1)
        while not rospy.is_shutdown():
            #print(message_arrived)
            hello=String();
            if head_message_arrived:
                print("message_arrived")
                if (head_message_arrived.button==1 and head_message_arrived.button==1):
                    #hello=Twist();
                    hello.data="Pepper is a good boy, you know"
                    print(message_arrived)
            message_arrived=False
            pub.publish(hello);

            rospy.sleep(0.5)
        #talker3()

    except rospy.ROSInterruptException:
        pass
