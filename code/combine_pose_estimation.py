import cv2
import os
import csv
import numpy as np

protoFile = r"E:\Drive D\MA-ICT Convergence\Semester 4\Human-Human-Interaction\src\utils\pose_deploy_linevec.prototxt"
weightsFile = r"E:\Drive D\MA-ICT Convergence\Semester 4\Human-Human-Interaction\src\utils\pose_iter_440000 (1).caffemodel"

nPoints = 18
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
                    'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 
                    'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16]]

mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], 
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], 
          [47,48], [49,50], [53,54], [51,52], [55,56], 
          [37,38], [45,46]]

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def getKeypoints(probMap, threshold=0.1):
    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)
    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []
    
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints

def detect_keypoints_and_draw_skeletons(image, net, csv_writer, image_filename):
    frameWidth = image.shape[1]
    frameHeight = image.shape[0]
    
    inHeight = 368
    inWidth = int((inHeight/frameHeight)*frameWidth)
    
    inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    
    net.setInput(inpBlob)
    
    output = net.forward()
    detected_keypoints = []
    keypoints_list = np.zeros((0,3))
    keypoint_id = 0
    threshold = 0.1
    
    for part in range(nPoints):
        probMap = output[0,part,:,:]
        probMap = cv2.resize(probMap, (image.shape[1], image.shape[0]))
        keypoints = getKeypoints(probMap, threshold)
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1
        
        detected_keypoints.append(keypoints_with_id)
    
    frameClone = image.copy()
    for i in range(nPoints):
        for n in range(len(detected_keypoints)):
            if len(detected_keypoints[n]) > i: 
                index = detected_keypoints[n][i][3]
                if index != -1:
                    cv2.circle(frameClone, detected_keypoints[n][i][:2], 3, [0,0,255], -1, cv2.LINE_AA)
    
    for i in range(17):
        for n in range(len(detected_keypoints)):
            index_A, index_B = POSE_PAIRS[i]
            if len(detected_keypoints[n]) > index_A and len(detected_keypoints[n]) > index_B:
                index = detected_keypoints[n][index_A][3], detected_keypoints[n][index_B][3]
                if -1 not in index:
                    B = np.int32(keypoints_list[index[1], 0])
                    A = np.int32(keypoints_list[index[0], 1])
                    cv2.line(frameClone, (B, A), (B, A), colors[i], 3, cv2.LINE_AA)
                    
    for keypoints in detected_keypoints:
        if keypoints:  
            for keypoint in keypoints:
                x, y, confidence, keypoint_id = keypoint  
                csv_writer.writerow([image_filename, x, y, confidence])
    
    return frameClone


net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


main_folder = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\all_channels_output_frame"
output_folder = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\all_channels_output_keypoint"

# Iterate through interaction classes
interaction_classes = ['none_interaction', 'hugging', 'kicking', 'pointing', 'punching', 'pushing']
for interaction_class in interaction_classes:
    interaction_class_folder = os.path.join(main_folder, interaction_class)
    
    # Iterate through camera channels
    camera_channels = ['c1', 'c2', 'c3', 'c4']
    for camera_channel in camera_channels:
        camera_channel_folder = os.path.join(interaction_class_folder, camera_channel)
        images = load_images_from_folder(camera_channel_folder)
        
        # Create a CSV file for each camera channel
        csv_filename = f"{interaction_class}_{camera_channel}_keypoints.csv"
        csv_filepath = os.path.join(output_folder, csv_filename)
        with open(csv_filepath, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Image File', 'X', 'Y', 'Confidence'])
            
            # Iterate through images
            for i, image in enumerate(images):
                image_filename = f"{interaction_class}_{camera_channel}_{i}.jpg"
                result_image = detect_keypoints_and_draw_skeletons(image, net, csv_writer, image_filename)
                
                image_filepath = os.path.join(output_folder, image_filename)
                cv2.imwrite(image_filepath, result_image)
                print(f"Image saved: {image_filepath}")
                
                # Print the current value of i
                print(f"Current value of i: {i}")
