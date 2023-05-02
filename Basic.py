import cv2
import numpy as np
import face_recognition

imgElon=face_recognition.load_image_file('ImageBasics/Elon musk.jpg')
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest=face_recognition.load_image_file('ImageBasics/Test.jpg')
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest=face_recognition.face_locations(imgElon)[0]
encodeElon=face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

#less the distance more it will be good
results=face_recognition.compare_faces([encodeTest],encodeElon)
facedist=face_recognition.face_distance([encodeElon],encodeTest)
cv2.putText(imgTest, f'{results} {round(facedist[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)


print(results,facedist)


print(faceLoc)
cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Test',imgTest)
cv2.waitKey(0)