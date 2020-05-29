import numpy as np
from termcolor import colored
import timeit
import _thread
import imutils
import time
import cv2
import os
import signal
import sys
import mss
from pynput.mouse import Button, Controller, Listener

mouse = Controller()

if __name__ == "__main__":
    print("Do not run this file directly.")

def start(config):

    # Config
    YOLO_DIRECTORY = config.get('Yolo','Directory')
    CONFIDENCE = config.getfloat('Yolo','Confidence')
    THRESHOLD = config.getfloat('Yolo','Threshold')
    ACTIVATION_RANGE = config.getint('Sampling','ActivationRange')
    AIM_LOCK = False
    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([YOLO_DIRECTORY, config.get('Yolo','Labels')])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([YOLO_DIRECTORY, config['Yolo']['Weights']])
    configPath = os.path.sep.join([YOLO_DIRECTORY, config['Yolo']['Config']])

    # Wait for buffering
    time.sleep(0.4)

    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    print("[INFO] loading neural-network from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def on_move(x, y):
        pass

    def on_click(x, y, button, pressed):
        nonlocal AIM_LOCK
        if button == Button.right:
            AIM_LOCK = not AIM_LOCK
            print('{0}'.format( AIM_LOCK))

    def on_scroll(x, y, dx, dy):
        pass

    listener = Listener(
        on_move=on_move,
        on_click=on_click,
        on_scroll=on_scroll)
    listener.start()

    # Test for GPU support
    build_info = str("".join(cv2.getBuildInformation().split()))
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        cv2.ocl.useOpenCL()
        print(colored("[OKAY] OpenCL is working!", "green"))
    else:
        print(
            colored("[WARNING] OpenCL acceleration is disabled!", "yellow"))
    if "CUDA:YES" in build_info:
        print(colored("[OKAY] CUDA is working!", "green"))
    else:
        print(
            colored("[WARNING] CUDA acceleration is disabled!", "yellow"))

    print()
    
    W, H = None, None
    # loop over frames from the video file stream
    with mss.mss() as sct:
        # Handle Ctrl+C in terminal, release pointers
        def signal_handler(sig, frame):
            # release the file pointers
            print("\n[INFO] cleaning up...")
            listener.stop
            sct.close()
            cv2.destroyAllWindows()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)

        # Part of the screen to capture
        Wd, Hd = sct.monitors[1]["width"], sct.monitors[1]["height"]
        HALF_RANGE = ACTIVATION_RANGE/2
        monitor = { "top": (Hd/2) - HALF_RANGE, 
                    "left": (Wd/2) - HALF_RANGE, 
                    "width": ACTIVATION_RANGE, 
                    "height": ACTIVATION_RANGE}
        print("[INFO] loading screencapture device...")
        while "Screen capturing":
            if not AIM_LOCK:
                continue
            else:
                start_time = timeit.default_timer()
                # Get raw pixels from the screen, save it to a Numpy array
                frame = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

                # if the frame dimensions are empty, grab them
                if W is None or H is None:
                    (H, W) = frame.shape[: 2]

                frame = cv2.UMat(frame)

                # construct a blob from the input frame and then perform a forward
                # pass of the YOLO object detector, giving us our bounding boxes
                # and associated probabilities
                blob = cv2.dnn.blobFromImage(frame, 1 / 260, (150, 150),
                                            swapRB=False, crop=False)
                net.setInput(blob)
                layerOutputs = net.forward(ln)

                # initialize our lists of detected bounding boxes, confidences,
                # and class IDs, respectively
                boxes = []
                confidences = []
                classIDs = []

                # loop over each of the layer outputs
                for output in layerOutputs:
                    # loop over each of the detections
                    for detection in output:
                        # extract the class ID and confidence (i.e., probability)
                        # of the current object detection
                        scores = detection[5:]

                        # classID = np.argmax(scores)
                        # confidence = scores[classID]
                        classID = 0  # person = 0
                        confidence = scores[classID]

                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > CONFIDENCE:
                            # scale the bounding box coordinates back relative to
                            # the size of the image, keeping in mind that YOLO
                            # actually returns the center (x, y)-coordinates of
                            # the bounding box followed by the boxes' width and
                            # height
                            box = detection[0: 4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")

                            # use the center (x, y)-coordinates to derive the top
                            # and and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            # update our list of bounding box coordinates,
                            # confidences, and class IDs
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)

                # apply non-maxima suppression to suppress weak, overlapping
                # bounding boxes
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

                # ensure at least one detection exists
                if len(idxs) > 0:

                    # Find best player match
                    bestMatch = confidences[np.argmax(confidences)]

                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        # draw target dot on the frame
                        target_dot = {'x': x + w / 2,'y': y + h / 5}
                        cv2.circle(frame, (int(target_dot['x']), int(target_dot['y'])), 5, (0, 0, 255), -1)

                        # draw a bounding box rectangle and label on the frame
                        # color = [int(c) for c in COLORS[classIDs[i]]]
                        #cv2.rectangle(frame, (x, y),
                        #                (x + w, y + h), (0, 0, 255), 2)

                        text = "TARGET {}%".format(int(confidences[i] * 100))
                        cv2.putText(frame, text, (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if bestMatch == confidences[i]:
                            # translate aimbot window coordinates to monitor coordinates
                            mouseX = monitor["left"] + (target_dot['x'])
                            mouseY = monitor["top"]  + (target_dot['y'])
                            # Set pointer position
                            mouse.position = (mouseX, mouseY)
                            # Snipe target
                            mouse.click(Button.left, 2)

                cv2.imshow("Neural Net Vision (Pine)", frame)
                elapsed = timeit.default_timer() - start_time
                sys.stdout.write(
                    "\r{1} FPS with {0} MS interpolation delay \t".format(int(elapsed*1000), int(1/elapsed)))
                sys.stdout.flush()
                if cv2.waitKey(1) & 0xFF == ord('0'):
                    break

    # Clean up on exit
    signal_handler(0, 0)
