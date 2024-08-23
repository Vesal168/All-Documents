# import cv2
# import numpy as np

# protoFile = r"E:\Drive D\MA-ICT Convergence\Semester 4\Human-Human-Interaction\src\utils\pose_deploy_linevec.prototxt"
# weightsFile = r"E:\Drive D\MA-ICT Convergence\Semester 4\Human-Human-Interaction\src\utils\pose_iter_440000 (1).caffemodel"

# def getKeypoints(probMap, threshold=0.1):
#     mapSmooth = cv2.GaussianBlur(probMap, (3,3), 0, 0)
#     mapMask = np.uint8(mapSmooth > threshold)
#     keypoints = []

#     contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     for cnt in contours:
#         blobMask = np.zeros(mapMask.shape)
#         blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
#         maskedProbMap = mapSmooth * blobMask
#         _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
#         if maxVal > 0:
#             keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

#     return keypoints

# def detect_keypoints(image, net):
#     frameWidth = image.shape[1]
#     frameHeight = image.shape[0]

#     inHeight = 368
#     inWidth = int((inHeight/frameHeight)*frameWidth)

#     inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
#     net.setInput(inpBlob)

#     output = net.forward()
#     detected_keypoints = []
#     keypoints_list = np.zeros((0,3))
#     keypoint_id = 0
#     threshold = 0.1

#     for part in range(18):  # Iterate over the 18 keypoint parts
#         probMap = output[0, part, :, :]
#         probMap = cv2.resize(probMap, (frameWidth, frameHeight))
#         keypoints = getKeypoints(probMap, threshold)
#         detected_keypoints.append(keypoints)

#         frameClone = image.copy()

#     return frameWidth, frameHeight, detected_keypoints

# # Load the network
# net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# # Load an example image
# image = cv2.imread(r"E:\Drive D\MA-ICT Convergence\Thesis\Human-Human-Interaction\dataset\ch4-output-frames\hugging\hugging_1_0.jpg")

# # Detect keypoints in the image
# width, height, keypoints = detect_keypoints(image, net)

# # Print keypoints and their dimensions
# print("Width shape:", ())
# print("Height shape:", ())
# print("Keypoints shape:", (len(keypoints),))
# for kp in keypoints:
#     print("Keypoint shape:", (len(kp), 3))
