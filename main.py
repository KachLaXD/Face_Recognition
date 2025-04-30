import cv2
import face_recognition

personOneImage = cv2.imread("mainPeople/GiorgiKachlishvili.jpg")
personOneRGB = cv2.cvtColor(personOneImage, cv2.COLOR_BGR2RGB)
personOneEncode = face_recognition.face_encodings(personOneRGB)[0]

personTwoImage = cv2.imread("mainPeople/LadoMegrelidze.jpg")
personTwoRGB = cv2.cvtColor(personTwoImage, cv2.COLOR_BGR2RGB)
personTwoEncode = face_recognition.face_encodings(personTwoRGB)[0]

for i in range(1, 11):
    filename = f"unknownPeople/img{i}.jpg"
    unknownImage = cv2.imread(filename)
    if unknownImage is None:
        print(f"Could not read {filename}")
        continue

    unknownRGB = cv2.cvtColor(unknownImage, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(unknownRGB)

    unknownEncode = encodings[0]

    resultOne = face_recognition.compare_faces([personOneEncode], unknownEncode)[0]
    resultTwo = face_recognition.compare_faces([personTwoEncode], unknownEncode)[0]

    if resultOne:
        print(f"Image {i}: Matches Giorgi")
    elif resultTwo:
        print(f"Image {i}: Matches Lado")
    else:
        print(f"Image {i}: Matches no one")
