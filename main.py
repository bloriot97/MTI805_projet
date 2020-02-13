import cv2
import numpy as np
import time

MODE = "COCO"

if MODE is "MPI" :
    protoFile = "models/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "models/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
elif MODE is "COCO":
    protoFile = "models/coco/pose_deploy_linevec.prototxt"
    weightsFile = "models/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

threshold = 0.1

# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

def anotate_frame(frame):
    frame = np.copy(frame)

    imgHeight, imgWidth = frame.shape[:2]
    
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (imgWidth, imgHeight), (0, 0, 0), swapRB=False, crop=False)
    
    # # Set the prepared object as the input blob of the network
    net.setInput(inpBlob)

    out = net.forward()

    H = out.shape[2]
    W = out.shape[3]
    # Empty list to store the detected keypoints
    points = []
    for i in range(8):
        # confidence map of corresponding body's part.
        probMap = out[0, i, :, :]
    
        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
        # Scale the point to fit on the original image
        x = (imgWidth * point[0]) / W
        y = (imgHeight * point[1]) / H
    
        if prob > threshold :
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    print(points)

    line_pairs = [(0,1),(1,2),(2,3), (3,4), (1,5), (5,6), (6,7)] 

    for pt_A_id, pt_B_id in line_pairs:
        pt_A = points[pt_A_id]
        pt_B = points[pt_B_id]

        if point!=None and pt_A!=None :
            cv2.line(frame, pt_A, pt_B, (0, 255, 255), 3, lineType=cv2.LINE_AA)
    for i, pt in enumerate(points):
        if pt:
            (x,y) = pt
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    
    return frame
    
# frame = cv2.imread("./data/single.png")

# frame = anotate_frame(frame)
 
# cv2.imshow("Output-Keypoints",frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

input_source = "data/scene19-camera1.mov"
cap = cv2.VideoCapture(input_source)
hasFrame, frame = cap.read()

vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

while cv2.waitKey(1) < 0:
    t = time.time()
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv2.waitKey()
        break

    frame = anotate_frame(frame)

    cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 100), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)

    cv2.imshow('Output-Skeleton', frame)

    vid_writer.write(frame)


vid_writer.release()