#from gtts import gTTS
#from playsound import playsound

# from datetime import date,datetime
# import pandas as pd
# import threading

# recognized__students__list = list()
# date_today_compressed = date.today().strftime("%a-%d-%m-%y")
# date_today_detailed = date.today().strftime("%a-%d-%B-%Y")

# attendance__dir = "C:\\Users\\Rishabh Rajpurohit\\Documents\\majorP\\Attendance"

# sub__name = "default subject"
# cls__type = "lecture"
# faculty__name = "default faculty"
# cls__time = "9am"
# s__no = 1

# savedModelLocation = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorP\\code\\trained_face_model.npy'
# baseDir = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\res\\data'
# fn_haar = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\code\\haarcascade_frontalface_default.xml'
# fn_dir = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\res\\database'

# sound__file = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\testvoice.mp3'

# def play__sound(s):
#     mytext = s
#     language = 'en'
#     myobj = gTTS(text=mytext, lang=language, slow=False)
#     myobj.save(sound__file)
#     playsound(sound__file)
#     os.remove(sound__file)


# def TrainFromSavedPhotos():
#     persons = os.listdir(baseDir)
#     print("Fetching Data...")
#     #play__sound('fetching data')
#     for person in persons:
#         images = os.listdir(baseDir+"\\"+person)
#         count = 0
#         size = 4
#         fn_name = person
#         path = os.path.join(fn_dir, fn_name)
#         if not os.path.isdir(path):
#             os.mkdir(path)
#         (im_width, im_height) = (68, 68)
#         haar_cascade = cv2.CascadeClassifier(fn_haar)

#         for image in images:
#             pathOfImg = baseDir+"\\"+person+"\\"+image
#             im = cv2.imread(pathOfImg)
#             gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#             mini = cv2.resize(gray, ((int)(gray.shape[1] / size), (int)(gray.shape[0] / size)))
#             faces = haar_cascade.detectMultiScale(mini)
#             faces = sorted(faces, key=lambda x: x[3])
#             if faces:
#                 count=count+1
#                 face_i = faces[0]
#                 (x, y, w, h) = [v * size for v in face_i]
#                 face = gray[y:y + h, x:x + w]
#                 face_resize = cv2.resize(face, (im_width, im_height))
#                 pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path) if n[0]!='.' ]+[0])[-1] + 1
#                 cv2.imwrite('%s/%s.png' % (path, pin), face_resize)
#                 cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
#                 cv2.putText(im, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_TRIPLEX,2,(0, 255, 0))
#                 if(count>=20):
#                     break; 

#     print('Training...')
#     #play__sound('the model is now training')

#     # Create a list of images and a list of corresponding names
#     (images, lables, names, id) = ([], [], {}, 0)
#     for (subdirs, dirs, files) in os.walk(fn_dir):
#         for subdir in dirs:
#             names[id] = subdir
#             subjectpath = os.path.join(fn_dir, subdir)
#             for filename in os.listdir(subjectpath):
#                 path = subjectpath + '/' + filename
#                 lable = id
#                 images.append(cv2.imread(path, 0))
#                 lables.append(int(lable))
#             id += 1
#     (im_width, im_height) = (68, 68)

#     # Create a Numpy array from the two lists above
#     (images, lables) = [numpy.array(lis) for lis in [images, lables]]
#     trained_face_recognizer=lr.train_lbph(images)
#     print('done')
#     #play__sound('the model is now trained')
#     numpy.save(savedModelLocation,trained_face_recognizer)


# def TrainFromWebcam():
#     count = 0
#     size = 4
#     #play__sound('Please enter your name or enrollment number')
#     fn_name = input('Enter Your Enrollment Number: ') 
#     path = os.path.join(fn_dir, fn_name)
#     if not os.path.isdir(path):
#         os.mkdir(path)
#     (im_width, im_height) = (68, 68)
#     haar_cascade = cv2.CascadeClassifier(fn_haar)
#     webcam = cv2.VideoCapture(0)

#     print("--------------Ensure that the room is well lit--------------")
#     print("-----------------------Taking pictures----------------------")
#     #play__sound("Ensure the room is well lit and the distance between face and camera is not more than half a meter for better accuracy")
#     # The program loops until it has a few images of the face.

#     while count < 20:
#         (rval, im) = webcam.read()
#         gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         mini = cv2.resize(gray, ((int)(gray.shape[1] / size), (int)(gray.shape[0] / size)))
#         faces = haar_cascade.detectMultiScale(mini)
#         faces = sorted(faces, key=lambda x: x[3])
#         if faces:
#             face_i = faces[0]
#             (x, y, w, h) = [v * size for v in face_i]
#             face = gray[y:y + h, x:x + w]
#             face_resize = cv2.resize(face, (im_width, im_height))
#             pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path) if n[0]!='.' ]+[0])[-1] + 1
#             cv2.imwrite('%s/%s.png' % (path, pin), face_resize)
#             cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
#             cv2.putText(im, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_TRIPLEX,2,(0, 255, 0))
#             time.sleep(0.38)        
#             count += 1
        
            
#         #cv2.imshow('OpenCV', im)
#         key = cv2.waitKey(10)
#         if key == 27:
#             break
#     print(str(count) + " images taken and saved to " + fn_name +" folder in database ")
#     cv2.destroyAllWindows()
#     webcam.release()
#     size = 4
#     print('Training...')
#     #play__sound('the model is now training')
#     # Create a list of images and a list of corresponding names
#     (images, lables, names, id) = ([], [], {}, 0)
#     for (subdirs, dirs, files) in os.walk(fn_dir):
#         for subdir in dirs:
#             names[id] = subdir
#             subjectpath = os.path.join(fn_dir, subdir)
#             for filename in os.listdir(subjectpath):
#                 path = subjectpath + '/' + filename
#                 lable = id
#                 images.append(cv2.imread(path, 0))
#                 lables.append(int(lable))
#             id += 1
#     (im_width, im_height) = (68, 68)

#     # Create a Numpy array from the two lists above
#     (images, lables) = [numpy.array(lis) for lis in [images, lables]]
#     trained_face_recognizer=lr.train_lbph(images)
#     print('done')
#     #play__sound('the model is now trained')
#     numpy.save(savedModelLocation,trained_face_recognizer)

# if f'Attendance-{date_today_detailed}.csv' not in os.listdir(attendance__dir):
#     with open(f'Attendance/Attendance-{date_today_detailed}.csv','w') as f:
#         f.write('S-No,Enrollment-No,Time-Stamp(Hour:Min:Sec),Subject-Name,Class-Type,Faculty-Name,Class-Time')

# from train import TrainFromSavedPhotos,TrainFromWebcam

# savedModelLocation = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorP\\code\\trained_face_model.npy'
# baseDir = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\res\\data'
# fn_haar = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\code\\haarcascade_frontalface_default.xml'
# fn_dir = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\res\\database'


# sound__file = 'C:\\Users\\Rishabh Rajpurohit\\Documents\\majorp\\testvoice.mp3'
# def play__sound(s):
#     mytext = s
#     language = 'en'
#     myobj = gTTS(text=mytext, lang=language, slow=False)
#     myobj.save(sound__file)
#     playsound(sound__file)
#     os.remove(sound__file)



# def LoadModelAndRun():
#     s__no = 1
#     recognized__students__list.clear()
#     sub__name, cls__type, faculty__name, cls__time = input('''\nEnter the following Details:\nFormat: <Sub_name>,<Class_type>,<Faculty_Name>,<Class_time>\n--->''').split(',')
#     try:
#         trained_face_recognizer=numpy.load(savedModelLocation)
#     except:
#         print('\n\nTrain the model first!')
#         return
#     # Load prebuilt model for Frontal Face
#     (im_width, im_height) = (68, 68)
#     # Part 2: Use fisherRecognizer on camera stream
#     (images, lables, names, id) = ([], [], {}, 0)
#     for (subdirs, dirs, files) in os.walk(fn_dir):
#         for subdir in dirs:
#             names[id] = subdir
#             subjectpath = os.path.join(fn_dir, subdir)
#             for filename in os.listdir(subjectpath):
#                 path = subjectpath + '/' + filename
#                 lable = id
#                 images.append(cv2.imread(path, 0))
#                 lables.append(int(lable)) 
#             id += 1

#     face_cascade = cv2.CascadeClassifier(fn_haar)
#     webcam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#     while True:
#         (_, im) = webcam.read()
#         gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#         for (x,y,w,h) in faces:
#             cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
#             face = gray[y:y + h, x:x + w]
#             face_resize = cv2.resize(face, (im_width, im_height))
#             prediction=lr.predict_lbph(face_resize,trained_face_recognizer,lables)
#             cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
#             if (prediction[1])<=100 and (prediction[1])>85:
#                 current__time = datetime.now().strftime('%H:%M:%S')
#                 df = pd.read_csv(f'Attendance/Attendance-{date_today_detailed}.csv')
#                 if(recognized__students__list.count(names[prediction[0]])==0):
#                     recognized__students__list.append(names[prediction[0]])
#                     with open(f'Attendance/Attendance-{date_today_detailed}.csv', 'a') as f:
#                         f.write(f'\n{s__no},{names[prediction[0]]},{current__time},{sub__name},{cls__type},{faculty__name},{cls__time}')
#                         s__no += 1
#                     print('%s - %s' % (names[prediction[0]],"marked PRESENT"))
#                     cv2.putText(im,'%s - %.0f%s' % (names[prediction[0]],prediction[1],"%"),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
#                     #play__sound(str(names[prediction[0]]) + "marked PRESENT")
#                     #play__sound("next student, please come forward")
                
#             else:
#                 cv2.putText(im,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

#     #cv2.imshow('OpenCV', im)
#         key = cv2.waitKey(10)
#         if key == 27:
#             break
#     cv2.destroyAllWindows()


# def init():
#     while True:
#         x = threading.Thread(target=playsound,args=("C:\\Users\\Rishabh Rajpurohit\\Documents\\majorP\\res\\menu__sound.mp3",))
#         x.start()
#         print('___________________________________________\n\n0.Train from Saved Photos\n1.Train From Webcam \n2.Run \n3.Reset Model \n4.Exit:\n')
#         user=input("-->")
#         x.join()
#         if user == '0':
#             TrainFromSavedPhotos()
            
#         elif user == '1':
#             TrainFromWebcam()
            
#         elif user=='2':
#             try:
#                 LoadModelAndRun()
#             except:
#                 break
#             finally:
#                 init()
    
#         elif user=='3':
#             os.remove(savedModelLocation)

#         elif user == '4':
#             print("Exiting!")
#             break
#         else:
#             print("Enter Valid input!\n\n")
#             #play__sound("Please enter a valid input")
        

# init()