
import os
import argparse

import cv2
import numpy as np

import dlib


class SmileDetector(object):
	"""class for SmileDetector"""
	def __init__(self, shape_model_path, smile_overlay_path, filter_size=10, win_perc=0.2,
		webcam_idx=0):
		super(SmileDetector, self).__init__()
		self.face_detector = dlib.get_frontal_face_detector()
		self.shape_detector = dlib.shape_predictor(shape_model_path)

		self.video_capture = cv2.VideoCapture(webcam_idx)	

		# Post-process filter for smile detection
		self.win_filter = []
		self.filter_size = filter_size
		self.win_perc = win_perc

		# Display smile :)
		self.smile_overlay = cv2.resize(cv2.imread(smile_overlay_path), (50, 50))


	def rect2tuple(self, rect):
		# Convert rectangle class from dlib to tuples
		x = rect.left()
		y = rect.top()
		w = rect.right() - x
		h = rect.bottom() - y

		return (x, y, w, h)


	def shape2np(self, shape):
		# Converts dlib 98 points into numpy array
		coords = np.zeros((68, 2)).astype(np.int)
	
		# loop over the 68 facial landmarks and convert them
		# to a 2-tuple of (x, y)-coordinates
		for i in range(0, 68):
			coords[i] = (shape.part(i).x, shape.part(i).y)
	
		return coords


	def get_smile(self, shape, bbox_face):
		# Get score for presence of smile in face by computing the
		# normalized norm of length of mouth
		(x, y, _, _) = bbox_face

		shape[:,0] -= x
		shape[:,1] -= y

		max_x = np.max(shape[:,0])
		max_y = np.max(shape[:,1])

		left_pt = np.array([shape[55-1,0]/max_x, shape[55-1,1]/max_y])
		right_pt = np.array([shape[49-1,0]/max_x, shape[49-1,1]/max_y])
		
		len_mouth = np.linalg.norm(left_pt - right_pt)

		return len_mouth


	def process_frame(self, frame):
		# TODO: check for no face
		
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# Detect face
		rects = self.face_detector(gray_frame, 1)
		
		for rect in rects:
			shape = self.shape_detector(gray_frame, rect)
			shape = self.shape2np(shape)

			bbox_face = self.rect2tuple(rect)
			(x, y, w, h) = bbox_face

			# get smile score
			smile_score = self.get_smile(shape.copy(), bbox_face)	

			# Post-process filter
			if smile_score > 0.40: self.win_filter.append(1)
			else: self.win_filter.append(0)
			if len(self.win_filter) >= self.filter_size:
				self.win_filter.pop(0)

			# Decide if smiling or not
			if np.sum(np.array(self.win_filter)) > self.win_perc:
				SMILE = True
			else:
				SMILE = False

			# Display | GUI
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
			
			for (x, y) in shape:
				cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

			if SMILE:
				# cv2.putText(frame, "SMILE", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
				# 	(0, 255, 0), 2)
				frame = overlay_transparent(frame, self.smile_overlay, 20, 20)

		return frame

	def run(self):
		# Starts running and processing streams from webcam
		while True:
			ret, frame = self.video_capture.read()

			if not ret:
				continue

			frame = cv2.resize(frame, (480, 380))

			frame = self.process_frame(frame)

			# Display | GUI
			cv2.imshow('Webcam Stream', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		
		# Cleaning
		self.video_capture.release()
		cv2.destroyAllWindows()


def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background


def main(webcam_idx):
	
	shape_model_path = './models/shape_predictor_68_face_landmarks.dat'
	smile_overlay_path = './data/smile.png'

	smile_detector = SmileDetector(shape_model_path, smile_overlay_path, webcam_idx=webcam_idx)

	# Start
	smile_detector.run()



if __name__ == "__main__":

	# Commandline args
	parser = argparse.ArgumentParser()
	parser.add_argument('--webcam_idx', type=int, default=0, help='Index/number \
		representing the webcam. If default does not work, please try other for your \
			system.')

	args = parser.parse_args()

	# Call the main method
	main(args.webcam_idx)