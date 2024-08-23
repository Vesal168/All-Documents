
# import cv2
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import csv

# protoFile = r"E:\Drive D\MA-ICT Convergence\Semester 4\Human-Human-Interaction\src\utils\pose_deploy_linevec.prototxt"
# weightsFile = r"E:\Drive D\MA-ICT Convergence\Semester 4\Human-Human-Interaction\src\utils\pose_iter_440000 (1).caffemodel"

# nPoints = 18
# keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
#                     'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 
#                     'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

# POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
#               [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
#               [1,0], [0,14], [14,16], [0,15], [15,17],
#               [2,17], [5,16]]

# mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], 
#           [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], 
#           [47,48], [49,50], [53,54], [51,52], [55,56], 
#           [37,38], [45,46]]

# colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
#          [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
#          [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]


# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder, filename))
#         if img is not None:
#             images.append(img)
#     return images

# def getKeypoints(probMap, threshold=0.1):
    
#     mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)
#     mapMask = np.uint8(mapSmooth>threshold)
#     keypoints = []
    
#     contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     for cnt in contours:
#         blobMask = np.zeros(mapMask.shape)
#         blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
#         maskedProbMap = mapSmooth * blobMask
#         _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
#         keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

#     return keypoints

# def getValidPairs(output):
#     valid_pairs = []
#     invalid_pairs = []
#     n_interp_samples = 10
#     paf_score_th = 0.1
#     conf_th = 0.7
    
#     for k in range(len(mapIdx)):
#         pafA = output[0, mapIdx[k][0], :, :]
#         pafB = output[0, mapIdx[k][1], :, :]
#         pafA = cv2.resize(pafA, (frameWidth, frameHeight))
#         pafB = cv2.resize(pafB, (frameWidth, frameHeight))

#         candA = detected_keypoints[POSE_PAIRS[k][0]]
#         candB = detected_keypoints[POSE_PAIRS[k][1]]
#         nA = len(candA)
#         nB = len(candB)

#         if( nA != 0 and nB != 0):
#             valid_pair = np.zeros((0,3))
#             for i in range(nA):
#                 max_j=-1
#                 maxScore = -1
#                 found = 0
#                 for j in range(nB):         
#                     d_ij = np.subtract(candB[j][:2], candA[i][:2])
#                     norm = np.linalg.norm(d_ij)
#                     if norm:
#                         d_ij = d_ij / norm
#                     else:
#                         continue
#                     interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
#                                             np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
#                     paf_interp = []
#                     for k in range(len(interp_coord)):
#                         paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
#                                            pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ]) 
#                     paf_scores = np.dot(paf_interp, d_ij)
#                     avg_paf_score = sum(paf_scores)/len(paf_scores)

#                     if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
#                         if avg_paf_score > maxScore:
#                             max_j = j
#                             maxScore = avg_paf_score
#                             found = 1

#                 if found:            
#                     valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

#             valid_pairs.append(valid_pair)
            
#         else: 
#             print("No Connection : k = {}".format(k))
#             invalid_pairs.append(k)
#             valid_pairs.append([])
#     return valid_pairs, invalid_pairs

# def getPersonwiseKeypoints(valid_pairs, invalid_pairs): 
#     personwiseKeypoints = -1 * np.ones((0, 19))

#     for k in range(len(mapIdx)):
#         if k not in invalid_pairs:
#             partAs = valid_pairs[k][:,0]
#             partBs = valid_pairs[k][:,1]
#             indexA, indexB = np.array(POSE_PAIRS[k])
#             for i in range(len(valid_pairs[k])): 
#                 found = 0
#                 person_idx = -1
#                 for j in range(len(personwiseKeypoints)):
#                     if personwiseKeypoints[j][indexA] == partAs[i]:
#                         person_idx = j
#                         found = 1
#                         break

#                 if found:
#                     personwiseKeypoints[person_idx][indexB] = partBs[i]
#                     personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

#                 elif not found and k < 17:
#                     row = -1 * np.ones(19)
#                     row[indexA] = partAs[i]
#                     row[indexB] = partBs[i]
#                     row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
#                     personwiseKeypoints = np.vstack([personwiseKeypoints, row])
#     return personwiseKeypoints

# def detect_keypoints_and_draw_skeletons(image, net, csv_writer, image_filename):
#     frameWidth = image.shape[1]
#     frameHeight = image.shape[0]

#     inHeight = 368
#     inWidth = int((inHeight/frameHeight)*frameWidth)

#     inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (inWidth, inHeight),
#                               (0, 0, 0), swapRB=False, crop=False)

#     net.setInput(inpBlob)

#     output = net.forward()
#     detected_keypoints = []
#     keypoints_list = np.zeros((0,3))
#     keypoint_id = 0
#     threshold = 0.1

#     for part in range(nPoints):
#         probMap = output[0,part,:,:]
#         probMap = cv2.resize(probMap, (image.shape[1], image.shape[0]))
#         keypoints = getKeypoints(probMap, threshold)
#         keypoints_with_id = []
#         for i in range(len(keypoints)):
#             keypoints_with_id.append(keypoints[i] + (keypoint_id,))
#             keypoints_list = np.vstack([keypoints_list, keypoints[i]])
#             keypoint_id += 1

#         detected_keypoints.append(keypoints_with_id)

#         frameClone = image.copy()
#     for i in range(nPoints):
#         for n in range(len(detected_keypoints)):
#             if len(detected_keypoints[n]) > i: # Check if keypoints exist for this part
#                 index = detected_keypoints[n][i][3]
#                 if index != -1:
#                     cv2.circle(frameClone, detected_keypoints[n][i][:2], 3, [0,0,255], -1, cv2.LINE_AA)

#     for i in range(17):
#         for n in range(len(detected_keypoints)):
#             index_A, index_B = POSE_PAIRS[i]
#             if len(detected_keypoints[n]) > index_A and len(detected_keypoints[n]) > index_B:
#                 index = detected_keypoints[n][index_A][3], detected_keypoints[n][index_B][3]
#                 if -1 not in index:
#                     B = np.int32(keypoints_list[index[1], 0])
#                     A = np.int32(keypoints_list[index[0], 1])
#                     cv2.line(frameClone, (B, A), (B, A), colors[i], 3, cv2.LINE_AA)
                    
#     for keypoints in detected_keypoints:
#         if keypoints:  # Check if keypoints are detected
#             for keypoint in keypoints:
#                 x, y, confidence, keypoint_id = keypoint  # Unpack the keypoint tuple
#                 csv_writer.writerow([image_filename, x, y, confidence])

#     return frameClone

# # Load the network
# net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# # Load images from folders
# main_folder = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\ch4-output-frames"
# subfolders = ['none-interaction', 'hugging', 'kicking', 'pointing', 'punching', 'pushing']
# output_folder = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\ch4_output_keypoints_csv"


# # Iterate through subfolders
# for subfolder in subfolders:
#     folder_path = os.path.join(main_folder, subfolder)
#     images = load_images_from_folder(folder_path)
    
#     # Create a CSV file for each subfolder
#     csv_filename = f"{subfolder}_keypoints.csv"
#     csv_filepath = os.path.join(output_folder, csv_filename)
#     with open(csv_filepath, mode='w', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
#         csv_writer.writerow(['Image File', 'X', 'Y', 'Confidence'])
        
#         # Iterate through images
#         for i, image in enumerate(images):
#             image_filename = f"{subfolder}_{i}.jpg"
#             result_image = detect_keypoints_and_draw_skeletons(image, net, csv_writer, image_filename)
            
#             image_filepath = os.path.join(output_folder, image_filename)
#             cv2.imwrite(image_filepath, result_image)
#             print(f"Image saved: {image_filepath}")

#                 # Print the current value of i
#             print(f"Current value of i: {i}")




import cv2
import numpy as np
import os
import csv

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
            images.append((filename, img))  # Store both filename and image
    return images

def getKeypoints(probMap, threshold=0.1):
    mapSmooth = cv2.GaussianBlur(probMap, (3,3), 0, 0)
    mapMask = np.uint8(mapSmooth > threshold)
    keypoints = []

    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        if maxVal > 0:
            keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints

def detect_keypoints(image, net, csv_writer, image_filename):
    frameWidth = image.shape[1]
    frameHeight = image.shape[0]

    inHeight = 368
    inWidth = int((inHeight/frameHeight)*frameWidth)

    inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)

    output = net.forward()
    detected_keypoints = []
    keypoints_list = np.zeros((0,3))
    keypoint_id = 0
    threshold = 0.1

    for part in range(nPoints):
        probMap = output[0, part, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))
        keypoints = getKeypoints(probMap, threshold)
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)

        frameClone = image.copy()

    for keypoints in detected_keypoints:
        if keypoints:  # Check if keypoints are detected
            for keypoint in keypoints:
                x, y, confidence, keypoint_id = keypoint  # Unpack the keypoint tuple
                csv_writer.writerow([image_filename, x, y, confidence])

# Load the network
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Load images from folders
main_folder = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\ch4-output-frames"
subfolders = ['none-interaction', 'hugging', 'kicking', 'pointing', 'punching', 'pushing']
output_folder = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\ch4_output_keypoints_csv"

# Iterate through subfolders
for subfolder in subfolders:
    folder_path = os.path.join(main_folder, subfolder)
    images = load_images_from_folder(folder_path)
    
    # Create a CSV file for each subfolder
    csv_filename = f"{subfolder}_keypoints.csv"
    csv_filepath = os.path.join(output_folder, csv_filename)
    with open(csv_filepath, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Image File', 'X', 'Y', 'Confidence'])
        
        # Iterate through images
        for image_filename, image in images:
            detect_keypoints(image, net, csv_writer, image_filename)
            print(f"Keypoints detected and saved for: {image_filename}")

            


