import cv2
import os
import logging
import numpy as np
from face_detection import FaceModel
from facial_landmarks_detection import FacialLandMarksModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseModel
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder

def build_argparser():

    parser = ArgumentParser()
    parser.add_argument("-f", "--facedetection", required=True, type=str,
                        help="Path to Face Detection model.")
    parser.add_argument("-fl", "--faciallandmark", required=True, type=str,
                        help="Path to Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headpose", required=True, type=str,
                        help="Path to Head Pose Estimation model.")
    parser.add_argument("-g", "--gazeestimation", required=True, type=str,
                        help="Path to Gaze Estimation model.")
    parser.add_argument("-i", "--input_model", required=True, type=str,
                        help="Specify Path to video file or enter cam for webcam")
    parser.add_argument("-flags", "--Flags", required=False, nargs='+',
                        default=[],
                        help="Specify the flags from fd, fld, hp, ge like --flags fd hp fld (Seperate each flag by space)"
                             "fd for Face Detection, fld for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation." )
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="path to CPU extension")
    parser.add_argument("-pt", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for model to detect the face accurately from the video frame.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Name of the device: "
                             "CPU, GPU, FPGA or MYRIAD")
    
    return parser

def model_assigner (args, logger):

    modelPathDict = {'FaceDetection':args.facedetection, 'FacialLandmarksDetection':args.faciallandmark, 
    'GazeEstimation':args.gazeestimation, 'HeadPoseEstimation':args.headpose}
    
    for fileNameKey in modelPathDict.keys():
        if not os.path.isfile(modelPathDict[fileNameKey]):
            logger.error("Unable to find specified "+fileNameKey+" xml file")
            exit(1)
            
    FDM = FaceModel(modelPathDict['FaceDetection'], args.device, args.prob_threshold, args.cpu_extension)
    FLDM = FacialLandMarksModel(modelPathDict['FacialLandmarksDetection'], args.device, args.prob_threshold, args.cpu_extension)
    GEM = GazeEstimationModel(modelPathDict['GazeEstimation'], args.device,args.prob_threshold, args.cpu_extension)
    HPEM = HeadPoseModel(modelPathDict['HeadPoseEstimation'], args.device, args.prob_threshold, args.cpu_extension)
    return FDM, FLDM, GEM, HPEM



def main():

    args = build_argparser().parse_args()
    Flags_ = args.Flags 
    
    logger = logging.getLogger()
    inputFilePath = args.input_model 
    inputFeeder = None
    if inputFilePath.lower()=="cam":
            inputFeeder = InputFeeder("cam")
    else:
        if not os.path.isfile(inputFilePath):
            logger.error("Unable to find specified video file")
            exit(1)
        inputFeeder = InputFeeder("video",inputFilePath)
    FDM, FLDM, GEM, HPEM =  model_assigner (args, logger) 

    
    mc = MouseController('medium','fast')
    
    inputFeeder.load_data()
    FDM.load_model()
    FLDM.load_model()
    HPEM.load_model()
    GEM.load_model()
    
    frame_count = 0
    logger.info(inputFeeder)
    for ret, frame in inputFeeder.next_batch():
        if not ret:
            break
        frame_count+=1
        if frame_count%5==0:
            cv2.imshow('video',cv2.resize(frame,(500,500)))
    
        key = cv2.waitKey(60)
        croppedFace, face_coords = FDM.predict(frame.copy(), args.prob_threshold)
        if type(croppedFace)==int:
            logger.error("Unable to detect the face.")
            if key==27:
                break
            continue
        
        hp_out = HPEM.predict(croppedFace.copy())
        
        left_eye, right_eye, eye_coords = FLDM.predict(croppedFace.copy())
        
        new_mouse_coord, gaze_vector = GEM.predict(left_eye, right_eye, hp_out)
        
        if (not len(Flags_)==0):
            preview_frame = frame.copy()
            if 'fd' in Flags_:
                preview_frame = croppedFace
            if 'fld' in Flags_:
                cv2.rectangle(croppedFace, (eye_coords[0][0]-10, eye_coords[0][1]-10), (eye_coords[0][2]+10, eye_coords[0][3]+10), (0,255,0), 3)
                cv2.rectangle(croppedFace, (eye_coords[1][0]-10, eye_coords[1][1]-10), (eye_coords[1][2]+10, eye_coords[1][3]+10), (0,255,0), 3)
                
            if 'hp' in Flags_:
                cv2.putText(preview_frame, "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(hp_out[0],hp_out[1],hp_out[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 0), 1)
            if 'ge' in Flags_:
                cv2.putText(frame,"Gaze Cords: x= {:.2f} , y= {:.2f} , z= {:.2f}".format(gaze_vector[0], gaze_vector[1], gaze_vector[2]),(20, 80),cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 0), 2)
                x, y, w = int(gaze_vector[0]*12), int(gaze_vector[1]*12),160
                le =cv2.line(left_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(le, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                re = cv2.line(right_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(re, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                croppedFace[eye_coords[0][1]:eye_coords[0][3],eye_coords[0][0]:eye_coords[0][2]] = le
                croppedFace[eye_coords[1][1]:eye_coords[1][3],eye_coords[1][0]:eye_coords[1][2]] = re
   
                
            cv2.imshow("visualization",cv2.resize(preview_frame,(500,500)))
        
        if frame_count%5==0:
            mc.move(new_mouse_coord[0],new_mouse_coord[1])    
        if key==27:
                break
    logger.error("VideoStream ended...")
    cv2.destroyAllWindows()
    inputFeeder.close()
     
    

if __name__ == '__main__':
    main() 
