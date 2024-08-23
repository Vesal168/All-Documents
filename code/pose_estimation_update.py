import cv2
import numpy as np
import os
import csv

protoFile = r"E:\Drive D\MA-ICT Convergence\Semester 4\Human-Human-Interaction\src\utils\pose_deploy_linevec.prototxt"
weightsFile = r"E:\Drive D\MA-ICT Convergence\Semester 4\Human-Human-Interaction\src\utils\pose_iter_440000 (1).caffemodel"

nPoints = 18

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append((filename, img)) 
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

def detect_keypoints(image, net):
    frameWidth = image.shape[1]
    frameHeight = image.shape[0]

    inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (frameWidth, frameHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)

    output = net.forward()
    detected_keypoints = []
    keypoints_list = np.zeros((0, 3))
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

    return keypoints_list

# Load the network
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Load images from folders
main_folder = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\ch4-output-frames"
subfolders = ['none-interaction', 'hugging', 'kicking', 'pointing', 'punching', 'pushing']
output_folder = r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\ch4_output_keypoints_numpy"

# Iterate through subfolders
for subfolder in subfolders:
    folder_path = os.path.join(main_folder, subfolder)
    images = load_images_from_folder(folder_path)

    # Create a folder for each subfolder
    subfolder_output_path = os.path.join(output_folder, subfolder)
    os.makedirs(subfolder_output_path, exist_ok=True)

    # Iterate through images
    for image_filename, image in images:
        keypoints_list = detect_keypoints(image, net)
        np.save(os.path.join(subfolder_output_path, f"{image_filename.split('.')[0]}.npy"), keypoints_list)
        print(f"Keypoints saved for: {image_filename}")
