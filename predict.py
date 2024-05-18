#importing necessary packages
from LBP.lbp import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
import os

#constructing the argument parser and parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--Training", required= True,
    help= "path to the the training images")
ap.add_argument("-eval", "--Testing", required= True,
    help = "path to the testing images")
args = vars(ap.parse_args())



desc = LocalBinaryPatterns(24,8)
data = []
labels = []


for imagePath in paths.list_images(args["Training"]):

    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)

    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)

model = LinearSVC(C = 100.0, random_state = 42)
model.fit(data, labels)

for imagePath in paths.list_images(args['testing']):


    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    prediction = model.predict(hist.reshape(1, -1))

    cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0,0, 255), 3)
    cv2.imshow("Image", image)
    cv2.waitkey(0)