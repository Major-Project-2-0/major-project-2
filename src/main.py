import os
from datetime import date,datetime
import cv2
from sklearn.neighbors import KNeighborsClassifier
import joblib
import warnings
import numpy as np

warnings.filterwarnings("ignore",category=DeprecationWarning)

recognized__students__list = list()
date_today_compressed = date.today().strftime("%a-%d-%m-%y")
date_today_detailed = date.today().strftime("%a-%d-%B-%Y")
recognized__students__list = []

sub__name = "default subject"
cls__type = "default class type"
faculty__name = "default faculty"
cls__time = "default time"

savedAttendanceLocation = r'C:\Users\rishabh\Documents\major-project-2\Attendance'
savedModelLocation = r'C:\Users\rishabh\Documents\major-project-2\static\face-recognition-model.pkl'
savedStaticLocation = r'C:\Users\rishabh\Documents\major-project-2\static'
savedHaarLocation = r'C:\Users\rishabh\Documents\major-project-2\haarcascade_frontalface_default.xml'
savedDatabaseLocation = r'C:\Users\rishabh\Documents\major-project-2\static\faces'
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### If these directories don't exist, create them
if not os.path.isdir(savedAttendanceLocation):
    os.makedirs(savedAttendanceLocation)
if not os.path.isdir(savedStaticLocation):
    os.makedirs(savedStaticLocation)
if not os.path.isdir(savedDatabaseLocation):
    os.makedirs(savedDatabaseLocation)
if f'Attendance-{datetoday}.csv' not in os.listdir(savedAttendanceLocation):
    with open(f'Attendance/Attendance-{datetoday}.csv','w',encoding="UTF-8") as f:
        f.write('S-no,Name,Enrollment,Time-Stamp(Hour:Min:Sec),Subject-Name,Class-Type,Faculty-Name,Class-Scheduled-Time')

def extract_faces(img):
    if img!=[]:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray,1.3,5)
        return face_points
    else:
        return []

def identify_face(facearray):
    model = joblib.load(savedModelLocation)
    return model.predict(facearray)

def train_model():
    if 'face-recognition-model.pkl' in os.listdir(savedStaticLocation):
        os.remove(savedModelLocation)
    faces=[]
    labels=[]
    userlist = os.listdir(savedDatabaseLocation)
    for user in userlist:
        for imgname in os.listdir(savedDatabaseLocation+f'\\{user}'):
            img = cv2.imread(f'{savedDatabaseLocation}\\{user}\\{imgname}')
            resized_face = cv2.resize(img,(50,50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,savedModelLocation)

def TrainFromWebcam():
    newusername = input('Enter Your Name: ')
    newuserid = input('Enter Enrollment Number: ') 
    userImageFolder = savedDatabaseLocation+'//'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userImageFolder):
        os.makedirs(userImageFolder)
    
    i,j=0,0
    cap=cv2.VideoCapture(0)
    ret = True
    while ret:
        ret,frame = cap.read()
        faces = extract_faces(frame)
        if faces is not None:
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,20),2)
                cv2.putText(frame,f'Images captured: {i}/50', (30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,20),2,cv2.LINE_AA)
                if j%10==0:
                    name = newusername+'_'+str(i)+'.jpg'
                    cv2.imwrite(userImageFolder+'/'+name,frame[y:y+h,x:x+w])
                    i+=1
                j+=1
            if j==500:
                break
            cv2.imshow('Adding New User',frame)
            if cv2.waitKey(1)==27:
                break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()


def startRecognition():
    if 'face-recognition-model.pkl' not in os.listdir(savedStaticLocation):
        print("Error: No Trained Model Present in Static Directory")
        return
    
    recognized__students__list.clear()
    sub__name, cls__type, faculty__name, cls__time = input('''\nEnter the following Details:\nFormat: <Sub_name>,<Class_type>,<Faculty_Name>,<Class_time>\n--->''').split(',')
    i=1
    cap=cv2.VideoCapture(0)
    ret = True
    while ret:
        ret,frame = cap.read()
        if extract_faces(frame)!=[]:
            (x,y,w,h) = extract_faces(frame)[0]
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1,-1))[0]
            identified_person_name = identified_person.split('_')[0]
            identified_person_id = identified_person.split('_')[1]

            #Add attendance functionality starts here
            cv2.putText(frame,f'{identified_person}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            current__time = datetime.now().strftime('%H:%M:%S')
            if(recognized__students__list.count(identified_person_id)==0):
                recognized__students__list.append(identified_person_id)
                with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                    f.write(f'\n{i},{identified_person_name},{identified_person_id},{current__time},{sub__name},{cls__type},{faculty__name},{cls__time}')
                print('%s - %s' % (identified_person,"marked PRESENT"))
                i=i+1
            #Add attendance functionality ends here
            
        cv2.imshow('Attendance',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()


def init():
    while True:
        print('SMART ATTENDANCE SYSTEM\n___________________________________________\n\n1.Add User to Database \n2.Run Recogntion \n3.Retrain Model \n4.Exit:\n')
        user=input("-->")
        if user == '1':
            TrainFromWebcam()
            
        elif user=='2':
            try:
                startRecognition()
            except Exception:
                break
            finally:
                init()
    
        elif user=='3':
            if 'face-recognition-model.pkl' in os.listdir(savedStaticLocation):
                os.remove(savedModelLocation)
            train_model()

        elif user == '4':
            print("Exiting!")
            break
        else:
            print("Enter Valid input!\n\n")
        

init()