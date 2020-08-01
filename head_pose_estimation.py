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

class HeadPoseModel:
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
        self.output_names = [o for o in self.network.outputs.keys()]
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
        input_dict = {self.input_name:img_processed}
        outputs = self.exec_net.infer(input_dict)
        output = self.preprocess_output(outputs)
        return output

    def check_model(self):
        pass

    def preprocess_input(self, image):
        image_resized = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        img_processed = np.transpose(np.expand_dims(image_resized,axis=0), (0,3,1,2))
        return img_processed

    def preprocess_output(self, outputs):
        outputs_ = []
        outputs_.append(outputs['angle_p_fc'].tolist()[0][0])
        outputs_.append(outputs['angle_r_fc'].tolist()[0][0])
        outputs_.append(outputs['angle_y_fc'].tolist()[0][0])
        return outputs_
