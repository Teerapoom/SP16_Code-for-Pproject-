# All import
import cv2
import csv
import collections
import numpy as np
from tracker import *

# Initialize Tracker เรียกใช้ class
tracker = EuclideanDistTracker()

# Initialize the videocapture object
cap = cv2.VideoCapture("./VDOdata/ThaiBuri_CMA13_7.30.mp4")
input_size = 320 #กำหนดขนาดทั้ง W H\
if not cap.isOpened():
    print("Failed to open video file")

# Detection confidence threshold ความถูกต้อง
confThreshold =0.7 # ค่าความเชื่อมั่น
nmsThreshold= 0.6 #(bounding box ที่ดีที่สุด)

font_color = (0, 0, 255)
font_size = 1
font_thickness = 3

# Middle cross line position ตำแหน่งเส้นขวางกลาง
middle_line_position =  420 #ขยับเส้น เดิม255
up_line_position = middle_line_position - 15 #เดิม15
down_line_position = middle_line_position + 15

# Store Coco Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')
#or code
    # className = []
    # with open("coco.names", "r") as f:
    #     className = [line.strip() for line in f.readlines()] #ข้อมูลใน list (รายการคลาส)

# class index for our required detection classes เลือกอันที่อยากตรวจจับ
required_class_index = [2, 3, 5, 7]
detected_classNames = []

## Model Files
modelConfiguration = 'yolov4.cfg'
modelWeigheights = 'yolov4.weights'

# configure the network model #ไว้อ่านค่าของ Configuration เเละ weights
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backend  (RUN NVIDIA GPU)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class สุ่มสีของclsass
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Function for finding the center of a rectangle ค้นหาจุดศูนย์กลางของBBox
# เพื่อหาระยะ x และระยะ y จากขอบด้านซ้ายและด้านบนของสี่เหลี่ยมผืนผ้าไปยังจุดศูนย์กลาง
# จำเป็นต้องแบ่งความกว้างและความสูงด้วย 2 เนื่องจากจุดศูนย์กลางของสี่เหลี่ยมผืนผ้าไม่ได้อยู่ที่มุมบนซ้าย แต่จะตั้งอยู่กึ่งกลางของความกว้างและความสูงของสี่เหลี่ยมผืนผ้า โดยการหารความกว้างและความสูงด้วย
#ตัวอย่างเช่น พิจารณาสี่เหลี่ยมผืนผ้าที่มีมุมบนซ้ายที่พิกัด (10, 10) กว้าง 50 และสูง 30 พิกัด x ของมุมบนซ้ายคือ 10 และครึ่งหนึ่งของความกว้าง (x1) คือ 25 การบวกค่าเหล่านี้เข้าด้วยกัน (10 + 25) จะได้พิกัด x ของจุดศูนย์กลาง ซึ่งเท่ากับ 35
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1 #หมายถึงระยะทางจากขอบด้านซ้ายของสี่เหลี่ยมผืนผ้าไปยังจุดศูนย์กลาง ในการหาพิกัด x จริงของจุดศูนย์กลาง เราต้องเพิ่มระยะนี้ให้กับพิกัด x ที่มุมบนซ้ายของสี่เหลี่ยมผืนผ้า (x) นี่ทำให้เราได้พิกัด x จริงของจุดศูนย์กลาง
    cy = y+y1
    return cx, cy


# List for store vehicle count information ข้อมูลการนับรถ
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]


# Function for count vehicle
def count_vehicle(box_id, img): #img เฟรม
    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection หาจุดศูนย์กลางของสี่เหลี่ยมเพื่อตรวจหา
    center = find_center(x, y, w, h)
    ix, iy = center

    # Find the current position of the vehicle ค้นหาตำแหน่งปัจจุบันของรถ
    if (iy > up_line_position) and (iy < middle_line_position):

        if id not in temp_up_list:
            temp_up_list.append(id)

    elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)

    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index] + 1

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1

    # Draw circle in the middle of the rectangle #วาด BBox
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    # print(up_list, down_list)


# Function for finding the detected objects from the network output
def postProcess(outputs,img): #เพื่อจับคู่ดัชนีคลาสที่ส่งคืนโดยโมเดล YOLO กับชื่อคลาสที่เกี่ยวข้อง จากนั้นกรองคลาสที่ไม่สนใจออก
    global detected_classNames #เพื่อให้เป็นตัวแปรกลางจะได้ไม่เพี๊ยน
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:] #ค่า Score หรือ confidences ที่ทำนายของวัตถุ
            classId = np.argmax(scores)  #ค่า Score สูงสุดวัตถุนั้น ตกอยู่คลาสอะไร
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w,h = int(det[2]*width) , int(det[3]*height) #จุดศูนย์กลางวัตถุ x,y
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds) --------------------------------------------------------------------------------------------
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            print(x,y,w,h)
            color = [int(c) for c in colors[classIds[i]]]
            name = classNames[classIds[i]]
            detected_classNames.append(name)
            # Draw classname and confidence score
            cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',
                  (x, y-10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # Draw bounding rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            detection.append([x, y, w, h, required_class_index.index(classIds[i])])
    #Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)
#
#
while (True):
    while(cap.isOpened()):
    # Capture frame-by-frame
        success, img = cap.read()
    # if not success:
    #      print("not pass")
    #      break
    #     img = cv2.resize(img,(input_size,input_size))
        ih, iw, channels = img.shape
    # print(iw,ih,channels)
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size,input_size), [0, 0, 0], 1, crop=False) #[0, 0, 0] ดังนั้นจึงไม่มีการลบค่าเฉลี่ย
#
    # Set the input of the network
        net.setInput(blob) #จัดรูปแบบในลักษณะที่เหมาะสมสำหรับเลเยอร์
        layersNames = net.getLayerNames() #ส่งคืนรายการสตริงที่แสดงชื่อของเลเยอร์
        outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
#   # Feed data to the network
        outputs = net.forward(outputNames)
#
    # Find the objects from the network output ค้นหาวัตถุจากเอาต์พุตเครือข่าย
        postProcess(outputs, img)


    # Draw the crossing lines

        cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
        cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 2)
        cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 2)
        #
        # Draw counting texts in the frame
        cv2.putText(img, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness) #115
        cv2.putText(img, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness) #160
        cv2.putText(img, "Car:        " + str(up_list[0]) + "     " + str(down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, font_color, font_thickness)
        cv2.putText(img, "Motorbike:  " + str(up_list[1]) + "     " + str(down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, font_color, font_thickness)
        cv2.putText(img, "Bus:        " + str(up_list[2]) + "     " + str(down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, font_color, font_thickness)
        cv2.putText(img, "Truck:      " + str(up_list[3]) + "     " + str(down_list[3]), (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
#
#     # Show the frames
        cv2.imshow('Output', img)
        key = cv2.waitKey(1)
        if key == ord("a"):
            break
#
    # Write the vehicle counting information in a file and save it
    # with open("data.csv", 'w') as f1:
    #     cwriter = csv.writer(f1)
    #     cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
    #     up_list.insert(0, "Up")
    #     down_list.insert(0, "Down")
    #     cwriter.writerow(up_list)
    #     cwriter.writerow(down_list)
    # f1.close()
    # print("Data saved at 'data.csv'")
    # Finally realese the capture object and destroy all active windows
cap.release()
cv2.destroyAllWindows()




