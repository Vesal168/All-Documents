import os
import sys
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

# set up GPU
tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']= '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ------------------Import OpenPose----------------------
# import OpenPose(Windows)
dir_path = os.path.dirname(os.path.realpath(__file__)) #e:\Project_Researchs\POST_PROJECTS\codes
print(dir_path)

# Chnage these variables to point to the correct folder
sys.path.append(dir_path + '/openpose/build/python/openpose/Release');

# Add external librabry(executable files) to system environment variable
os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/openpose/build/x64/Release;' +  dir_path + '/openpose/build/bin;'
import pyopenpose as op

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = dir_path + "/openpose/models/"

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

#----------------------------------------

def make_extractedKeypoint_dir(make_folder):
    main_folder = make_folder
    class_types = ["hugging", "kicking", "none-interaction", "pointing", "punching", "pushing"]

    if not os.path.isdir(main_folder):
        try:
            os.makedirs(main_folder)
            for class_folder in class_types:
                os.makedirs(os.path.join(main_folder, class_folder))
            print("Finished creating directories")

        except Exception as e:
            print("Error:", e)
    else:
        print("Directory already exists")


'''
datum = op.Datum()  
imageToProcess = cv2.imread("../../asset/Capture.jpg")
#print(imageToProcess)
datum.cvInputData = imageToProcess 
opWrapper.emplaceAndPop(op.VectorDatum([datum])) 
keypoint_ext = datum.poseKeypoints[:,:2]
print("extracted keypoints: ", keypoint_ext )
plt.imshow(datum.cvOutputData)
plt.show()
cv2.imshow("OpenPose 1.7.0", datum.cvOutputData)
key = cv2.waitKey(200)'''

#-----------------Extract keypoint frome each video--------------------------

creat_directory_folder = "../../dataset/keypoints_extracted_clean"
make_extractedKeypoint_dir(creat_directory_folder)

video_path = "../../dataset/combined_camera_video"
class_lists = os.listdir(video_path)

for each_class in range(0,len(class_lists)): 
    class_file_path = os.path.join(video_path,class_lists[each_class])
    file_lists = os.listdir(class_file_path)

    for file_name_dix in range(0,len(file_lists)): 
        file_name = file_lists[file_name_dix]
        file_path = os.path.join(class_file_path, file_name)

        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

        all_frames_keypoints = []
        one_object_detect = np.zeros((25,2))
        miss_one_object_detect = np.zeros((2,50,2))
        while True:
            datum = op.Datum()
            ret, frame = cap.read() 
            if not ret: 
                break
            resized_frame = cv2.resize(frame, (800, 500))
            imageToProcess = resized_frame 

            datum.cvInputData = imageToProcess 
            opWrapper.emplaceAndPop(op.VectorDatum([datum])) 
            print("datum.poseKeypoints: ", datum.poseKeypoints[:, :, :2].shape)
            print(":   ", datum.poseKeypoints.size)

            # check condiction to prevent OpenPose return None 
            if (datum.poseKeypoints is not None
                and datum.poseKeypoints.size ==75
                and datum.poseKeypoints.shape==(1,25,3)): 

                kp_extracted = (datum.poseKeypoints[0, :, :2], one_object_detect) 
                print("detect 1 objects!")

            elif (datum.poseKeypoints is not None
                  and datum.poseKeypoints.size ==150
                  and datum.poseKeypoints.shape==(2,25,3)):
                kp_extracted = datum.poseKeypoints[:2, :, :2]
                print("detect 2 objects!")
            
            elif datum.poseKeypoints is not None:
                kp_extracted = datum.poseKeypoints[:2, :, :2]
                print("detect multiple objects!")

            else: 
                kp_extracted = (one_object_detect, one_object_detect)
                print("cannot detect!")

            print("kp_extracted: ", kp_extracted)
            all_frames_keypoints.append(kp_extracted)

        # Release video capture object
        cap.release()
        arr_all_frames_keypoints =  np.array(all_frames_keypoints)

        # save extracted keypoints to folder
        class_types = ["hugging", "kicking", "none-interaction", "pointing", "punching", "pushing"]
        class_index = class_types.index(class_lists[each_class])
        edit_file_name = file_name.split(".")[0] + ".npy"
        save_extractedkey_path = os.path.join(creat_directory_folder, class_types[class_index], edit_file_name)
        np.save(save_extractedkey_path, arr_all_frames_keypoints)

        print("Finished extracted keypoints from:", file_path)


