'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
import numpy as np
import os
from openvino.inference_engine import IECore, IENetwork

class GazeEstimationModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU',threshold = 0.6 , extensions=None):
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.threshold = threshold
        self.model_structure = self.model_name
        self.model_weights = os.path.splitext(model_name)[0] + ".bin"
        self.core = IECore()
        self.network = self.core.read_network(model=self.model_structure, weights=self.model_weights)
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_names].shape
        self.exec_net = None
       

    def load_model(self):
       
        supported_layers = self.core.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers)!=0 and self.device=='CPU':
            print("unsupported layers found:{}".format(unsupported_layers))
            if self.extensions:
                self.core.add_extension(self.extensions, self.device)
                supported_layers = self.core.query_network(network = self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers)!=0:
                    print("Error:Unsupported layers still found:{}".format(unsupported_layers))
                    exit(1)
            else:
                print("Error: Exteion layers not found!")
                exit(1)
        self.exec_net = self.core.load_network(network=self.network, device_name=self.device,num_requests=1)
       
       

    def predict(self, left_image, right_image, hpa):
       
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        left_image_ = left_image
        right_image_ = right_image 
        left_img_, right_img_ = self.preprocess_input(left_image_ , right_image_, hpa)
        input_dict = {'head_pose_angles':hpa, 'left_eye_image':left_img_, 'right_eye_image':right_img_}
        outputs = self.exec_net.infer(input_dict)
        new_mouse_probs, gaze_vector = self.preprocess_output(outputs,hpa)
        return new_mouse_probs, gaze_vector


    def check_model(self):
        pass

    def preprocess_input(self, left_image, right_image):
        left_image_ = cv2.resize(left_image, (self.input_shape[3], self.input_shape[2]))
        right_image_ = cv2.resize(right_image, (self.input_shape[3], self.input_shape[2]))
        left_processed = np.transpose(np.expand_dims(left_image_,axis=0), (0,3,1,2))
        right_processed = np.transpose(np.expand_dims(right_image_,axis=0), (0,3,1,2))
        return left_processed, right_processed

    def preprocess_output(self, output, hpa):
        gaze_vector = outputs[self.output_names[0]].tolist()[0]
        rollValue = hpa[2]
        cos_ = math.cos(rollValue * math.pi / 180.0)
        sin_ = math.sin(rollValue * math.pi / 180.0)        
        newx = gaze_vector[0] * cos_ + gaze_vector[1] * sin_
        newy = -gaze_vector[0] *  sin_ + gaze_vector[1] * cos_
        new_coords = (newx,newy)
        return new_coords, gaze_vector
