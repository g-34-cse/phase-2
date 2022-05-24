# phase-2

#function for preprocessing image

def preprocess_img(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #convert bgr to rgb
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#convert to grayscale
    
    #identify the shape of image
    a,b=gray.shape
    
    #modify shape into *1/3
    new_a=a//3
    new_b=b//3
 
    resized=cv2.resize(gray,(new_b,new_a))
    return resized

#function for preprocessing of face cascade

def detect_face(img,face_cascade):
    faces=face_cascade.detectMultiScale(img, 1.1, 4)
    
    #Draw rectangle around the faces
    for(x, y, w, h) in faces: 
       cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    #identify shape of image
    a,b=img.shape


    #enlarge back to normal size
    img=cv2.resize(img,(b*3,a*3))
    return img

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# functions for using live webcam and detection

cam=cv2.VideoCapture(0)
while(True):
     result, image = cam.read() #read the camera input and save in a variable
     resized3 = preprocess_img(image) #1st function call
     detected_img = detect_face(resized3,face_cascade) #2nd function call
     cv2.imshow("image", detected_img)#display
     if cv2.waitkey (1) & 0xFF==ord ('q') : #keyboard character used for exiting webcam window
        cv2.destroyAllwindows() #all windows related to opencv are terminated
        break
cam.release()
