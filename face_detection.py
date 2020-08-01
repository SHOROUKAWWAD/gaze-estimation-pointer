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

class FaceModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU',threshold= 0.6 , extensions=None):
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
       
       

    def predict(self, image):
       
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        image_ = image
        img_processed = self.preprocess_input(image_)
        outputs = self.exec_net.infer({self.input_name:img_processed})
        coords = self.preprocess_output(outputs, prob_threshold)
        if (len(coords)==0):
            return 0, 0
        face = coords[0]
        height=image.shape[0]
        width=image.shape[1]
        face = face * np.array([width, height, width, height])
        face = face.astype(np.int32)        
        cropped_face = image[face[1]:face[3], face[0]:face[2]]
        return cropped_face, face


    def check_model(self):
        pass

    def preprocess_input(self, image):
        pre_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        pre_image = pre_image.transpose((2, 0, 1))
        pre_image = pre_image.reshape(1, *pre_image.shape)
        return pre_image

    def preprocess_output(self, outputs):
        probs =[]
        boxes = outputs[self.output_names][0][0]
        for box in boxes:
            conf = box[2]
            if conf> self.threshold:
                x_min=box[3]
                y_min=box[4]
                x_max=box[5]
                y_max=box[6]
                probs.append([x_min,y_min,x_max,y_max])
        return probs
