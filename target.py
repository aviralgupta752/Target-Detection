import imutils
import cv2
cap = cv2.VideoCapture(0)
while(True):
	ret, frame = cap.read()
	status = "No Targets"
	if(ret == False):
		break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (7,7), 0)
	edged = cv2.Canny(blurred, 75, 200)

	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx =cv2.approxPolyDP(c, 0.02*peri, True)
		if(len(approx) >=4 and len(approx) <= 6):
			(x,y,w,h) = cv2.boundingRect(approx)
			ratio = w / float(h)

			area = cv2.contourArea(c)
			hullArea = cv2.contourArea(cv2.convexHull(c))
			solidity = area / float(hullArea) 

			keepDims = w > 25 and h > 25
			keepSolidity = solidity > 0.9
			keepAspectRatio = ratio >=0.5 and ratio <= 1.5

			if keepDims and keepSolidity and keepAspectRatio:
				cv2.drawContours(frame, [approx], -1, (0,0,255), 4)
				status = "Target(s) Acquired"
				M = cv2.moments(approx)
				(cX, cY) = (int(M["m10"] // M["m00"]), int(M["m01"] // M["m00"]))

	cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
		(0, 0, 255), 2)
	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) & 255 == ord('q'):
		break

cv2.destroyAllWindows()

