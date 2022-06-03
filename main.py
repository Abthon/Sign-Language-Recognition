import pickle
from Imports  import *
################################################### <Constants> ##########################################################

FPS = 30
WIDTH = 1280
HEIGHT = 720
SCORE = 0
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Screen Settings
SCREEN_SIZE = pyautogui.size()
SCREEN_WIDTH = SCREEN_SIZE[0]
SCREEN_HEIGHT = SCREEN_SIZE[1]

# Camera object and its settings 
CAM = cv2.VideoCapture(1)
CAM.set(cv2.CAP_PROP_FRAME_WIDTH,WIDTH)
CAM.set(cv2.CAP_PROP_FRAME_HEIGHT,HEIGHT)
CAM.set(cv2.CAP_PROP_FPS,FPS)
CAM.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))


# Key Points 
keyHandJoints = [0,4,5,9,13,17,8,12,16,20]
#################################################### </Constants> ##############################################################



class myHands:
    import mediapipe as mp
    def __init__(self,static_image_mode=False, max_num_hands=1, model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        self.static_image_mode=static_image_mode    # STATIC_IMAGE_MODE: If set to false, the solution treats the input images as a video stream. 
        self.max_num_hands=max_num_hands            # Maximum number of hands to detect. Default to 2.
        self.model_complexity=model_complexity      # Complexity of the hand landmark model: 0 or 1
        self.min_detection_confidence=min_detection_confidence # Minimum confidence value ([0.0, 1.0]) from the hand detection model for the detection to be considered successful. Default to 0.5.
        self.min_tracking_confidence=min_tracking_confidence   # Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the hand landmarks to be considered tracked successfully,
        self.hands = self.mp.solutions.hands.Hands(self.static_image_mode,self.max_num_hands,self.model_complexity,self.min_detection_confidence,self.min_tracking_confidence)
        self.mpDraw = self.mp.solutions.drawing_utils


    def Marks(self,frame):
        myHands = list() # [[first_hand], [second_hand],[third_hand],[...],...]
        result = self.hands.process(frame)
        if (result.multi_hand_landmarks is not None):
            for hand in result.multi_hand_landmarks:
                myHand = list()
                for handLandMarks in hand.landmark:
                    myHand.append((int(handLandMarks.x * WIDTH),int(handLandMarks.y * HEIGHT)))
                myHands.append(myHand)
        return [myHands,result]

    # A function to draw connection between all landmark
    def drawConnection(self,rgbFrame,bgrFrame):
        result = self.Marks(rgbFrame)[1]
        if result.multi_hand_landmarks is not None:
            for hands in result.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(bgrFrame,hands,self.mp.solutions.hands.HAND_CONNECTIONS)



# A function to find a distance between the known and unknown gestures
def findDistance(handLandMarks):
    distanceMatrix = np.zeros([len(handLandMarks),len(handLandMarks)],dtype='float')
    distanceFromPoint0To9 = sqrt(pow((handLandMarks[0][0]-handLandMarks[9][0]),2) + pow((handLandMarks[0][1] - handLandMarks[9][1]),2))
    for row in range(0,len(handLandMarks)):
        for column in range(0,len(handLandMarks)):
            distanceMatrix[row][column] = sqrt(pow((handLandMarks[row][0]-handLandMarks[column][0]),2) + pow((handLandMarks[row][1] - handLandMarks[column][1]),2))/distanceFromPoint0To9

    return distanceMatrix



# A function to find our error value between the known and unknown value
def error(knownMatrix,unknownMatrix,keyPoints):
    error = 0
    for row in keyPoints:
        for column in keyPoints:
            error = error + abs(knownMatrix[row][column] - unknownMatrix[row][column])
    
    return int(error)


def findGesture(unknownGesture,knownGestures,gestureNames,tollerance,keyHandJoints):
    listOfErrorValues =  list()
    errorValue = 0
    gestureName = ''

    for i in range(0,len(gestureNames)):
        errorValue = errorValue + error(knownGestures[i],unknownGesture,keyHandJoints)
        listOfErrorValues.append(errorValue)
        errorValue = 0

    minError = min(listOfErrorValues)
    gestureName = gestureNames[listOfErrorValues.index(minError)] if minError <= tollerance else 'Unknown'

    return gestureName


def random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(3))


tollerance = 15
# knownGesture = None
errorValue = None
mpHands = myHands()
time.sleep(2)
train = int(input("Please Enter 1 to train or 0 to Recognize? "))
if train == 1:
    counter = 0
    knownGestures = []
    numberOfGastures = int(input("Enter the number of Gesture's you went to train: "))
    gestureNames = []

    for i in range(1,numberOfGastures+1):
        prompt = f"Type gesture number {i} # "
        gestureName = input(prompt)
        gestureNames.append(gestureName)

    trainName = input("File for training data? (Enter Enter or Default): ")
    if trainName == '':
        trainName = "default"
    trainName = trainName + ".pkl"

if train == 0:
    trainName = input("Please enter your dataSet name, (Enter for default): ")
    if trainName == '':
        trainName = 'default'
    trainName = trainName + '.pkl'
    with open(trainName,'rb') as f:
        gestureNames = pickle.load(f)
        knownGestures = pickle.load(f)


print(knownGestures)



while CAM.isOpened():
    isNotEmpty,frame = CAM.read()

    frame = cv2.flip(frame,1)
    rgbFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    if not(isNotEmpty):
        break

    myHands = mpHands.Marks(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    if train == 1:
        if myHands[0] != []:
            print(f"Press t to train Ab's model what a {gestureNames[counter]} looks like ")
            if cv2.waitKey(1) == ord('t'):              
                knownGesture = findDistance(*myHands[0])
                knownGestures.append(knownGesture)
                counter += 1
                if counter >= len(gestureNames):
                    isTraning = 0
                    with open(trainName,'wb') as f:
                        pickle.dump(gestureNames,f)
                        pickle.dump(knownGestures,f)
    
    if train == 0:
        if myHands[0] != []:
            unknownGesture = findDistance(*myHands[0])
            # errorValue = error(knownGesture,unknownGesture,keyHandJoints)
            DetectedGesturesName = findGesture(unknownGesture,knownGestures,gestureNames,tollerance,keyHandJoints)
            cv2.rectangle(frame, (10,10),(WIDTH-10,150),(255,0,255),-1)
            cv2.putText(frame,f'DetectedSign: {DetectedGesturesName}',(50,100),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),3)

    if myHands[0] !=[]:
        for val in keyHandJoints:
            # cv2.putText(frame,str(val), myHands[0][0][val],cv2.FONT_HERSHEY_COMPLEX,0.8,(255,0,255,),2)
            cv2.circle(frame,myHands[0][0][val],13,random_color(),-1)


    mpHands.drawConnection(rgbFrame,frame)
    cv2.imshow('Camera',frame)
    myHands
    if cv2.waitKey(1) == ord('q'):
        break

CAM.release()




    

