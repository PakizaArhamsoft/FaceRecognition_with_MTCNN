import os
from utils import Detector
import cv2

# Set directories
BASE_DIR = "test_images"
list_imgs = os.listdir("test_images")

# create detector object
detector = Detector()

for im in list_imgs:
	img = cv2.imread(os.path.join(BASE_DIR, im))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img,(460,460))
	# get predictions and draw them on image
	predictions = detector.get_people_names(img, speed_up=False, downscale_by=1)
	annoted_image = detector.draw_results(img, predictions)

	image = cv2.cvtColor(annoted_image, cv2.COLOR_RGB2BGR)
	cv2.imwrite(f"Detect_images/{im.split('.')[0]}_infered.png", image)
	cv2.imshow("image", image)
	cv2.waitKey(3)
