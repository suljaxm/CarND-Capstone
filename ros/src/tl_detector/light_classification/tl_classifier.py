from styx_msgs.msg import TrafficLight
import os
import numpy as np
import tensorflow as tf
import cv2
import rospy

class TLClassifier(object):
    def __init__(self, is_site):
        #TODO load classifier
        self.is_site = is_site

        rospy.logwarn("is_site is %s!", self.is_site)

        self.image_tensor = None
        self.detection_boxes = None
        self.detection_scores = None
        self.detection_classes = None
        
        self.detection_graph = None

        self.load_model()
        
        self.category_index = {1:'green', 2:'red', 3:'yellow', 4:'unknown'}

    def load_model(self):
        # get the path where saved the models
        dir = os.path.dirname(os.path.realpath(__file__))
        path_real = dir + '/models/model_ssd/frozen_inference_graph.pb'
        path_sim = dir + '/models/frozen_inf_graph_sim_ssd.pb'
        graph_file = path_real if self.is_site else path_sim
        # load frozen graph
        """Loads a frozen inference graph"""
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')
    
        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')          

        rospy.logwarn("load model is ok!")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        detected_light = TrafficLight.UNKNOWN
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), axis=0)
        with tf.Session(graph=self.detection_graph) as sess:
            boxes, scores, classes = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                            feed_dict={self.image_tensor: image_np})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)


        if scores[scores.argmax()] > 0.4:
            color = self.category_index[classes[scores.argmax()]]
            rospy.logwarn("color is %s", color)
            if color == 'green':
                detected_light = TrafficLight.GREEN
            elif color == 'red':
                detected_light = TrafficLight.RED
            elif color == 'yellow':
                detected_light = TrafficLight.YELLOW
        else:
            detected_light = TrafficLight.UNKNOWN


        return detected_light
