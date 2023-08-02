import cv2, os, face_recognition
import numpy as np

path = "/Users/arnavg/PythonProjects/personal/CV/facialRec/dataSet"


def store(folder):
    images = []
    names = []
    for file in os.listdir(folder):
        vals = file.split(".")
        name = vals[0]
        names.append(name)
        img = cv2.imread(os.path.join(folder, file))
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if rgb_img is not None:
            images.append(rgb_img)
    return images, names


def encode(imgs):
    encoded = []
    for im in imgs:
        img_encoding = face_recognition.face_encodings(im)[0]
        encoded.append(img_encoding)
    return encoded


def detect(frame, encoded, names):
    small_frame = cv2.resize(frame, (0, 0), fx=0.15, fy=0.15)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locs = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locs)

    values = encodings(face_encodings, encoded, names)

    face_locations = np.array(face_locs)
    face_locations = face_locations / 0.15
    return face_locations.astype(int), values


def encodings(encodings, encoded, names):
    values = []
    for face_encoding in encodings:
        matches = face_recognition.compare_faces(encoded, face_encoding)
        name = "no data"

        face_distances = face_recognition.face_distance(encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = names[best_match_index]
        values.append(name)
    return values


def main():
    vals, names = store(path)
    encoded = encode(vals)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        face_locations, face_names = detect(frame, encoded, names)

        for face_loc, name in zip(face_locations, face_names):
            # print(face_loc)
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(
                frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        if not ret:
            print("Failed to capture frame")
            break

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


main()
