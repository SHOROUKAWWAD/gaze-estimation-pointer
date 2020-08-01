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

class FacialLandMarksModel:
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
        img_processed = self.preprocess_input(image.copy())
        input_dict = {self.input_name:img_processed}
        outputs = self.exec_net.infer(input_dict)
        probs = self.preprocess_output(outputs)
        h=image.shape[0]
        w=image.shape[1]
        probs = probs* np.array([w, h, w, h])
        probs = probs.astype(np.int32) 
        left_xmin=probs[0]-10
        left_ymin=probs[1]-10
        left_xmax=probs[0]+10
        left_ymax=probs[1]+10
        
        right_xmin=probs[2]-10
        right_ymin=probs[3]-10
        right_xmax=probs[2]+10
        right_ymax=probs[3]+10
        
        left_eye =  image[left_ymin:left_ymax, left_xmin:left_xmax]
        right_eye = image[right_ymin:right_ymax, right_xmin:right_xmax]
        eye_probs = [[left_xmin,left_ymin,left_xmax,left_ymax], [right_xmin,right_ymin,right_xmax,right_ymax]]
        return left_eye, right_eye, eye_probs
        
    def check_model(self):
        ''

    def preprocess_input(self, image):
        image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_cvt, (self.input_shape[3], self.input_shape[2]))
        img_processed = np.transpose(np.expand_dims(image_resized,axis=0), (0,3,1,2))
        return img_processed
            

    def preprocess_output(self, outputs):
        outs = outputs[self.output_names][0]
        leye_x = outs[0].tolist()[0][0]
        leye_y = outs[1].tolist()[0][0]
        reye_x = outs[2].tolist()[0][0]
        reye_y = outs[3].tolist()[0][0]
        
        return (leye_x, leye_y, reye_x, reye_y)
