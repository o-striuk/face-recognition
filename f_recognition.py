import face_recognition as fr
import os
import cv2
import numpy as np

FRAME_THICKNESS = 2
FONT_THICKNESS = 1

def encode_faces():
    """
    Goes through the folder with face samples and encodes them,
    returns a dictionary with the 'key: value' pair — name and encoded image.
    """
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./samples"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("samples/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


def encode_unknown_image(img):
    """
    Encode a face sample given the file name.
    """
    face = fr.load_image_file("samples/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def classify_sample(im):
    """
    Will find all the faces in a provided test image and label them if it can
    identify them. Returns list of face names.
    """
    faces = encode_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)

    face_locations = fr.face_locations(img)
    unknown_face_encodings = fr.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # Check if the face matches the known face/faces.
        matches = fr.compare_faces(faces_encoded, face_encoding)
        name = "unknown"

        # Given a list of face encodings, compare them to a known face encoding,
        # and get an Euclidean distance for each comparison face. The distance
        # tells you how similar the faces are.
        face_distances = fr.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draws an identification frame around the face.
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), FRAME_THICKNESS)

            # Draws a label with a name under the identified face/faces.
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), FONT_THICKNESS)


    # Displays the test image.
    while True:

        cv2.imshow("Test Image (press 'q' to exit)", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names


print(classify_sample("test.jpg"))
