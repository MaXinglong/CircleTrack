import datetime
import cv2

import track


video = cv2.VideoCapture('../output_multi_long_266.avi')
# video = cv2.VideoCapture('../output_two.avi')

solver = track.Solver()
n = 1000

ret, frame = video.read()

while ret:
    out = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th, out = cv2.threshold(out, 50, 255, cv2.THRESH_BINARY)
    out, contours, _ = cv2.findContours(out, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    measurements = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        center = [x+w/2, y+h/2]
        measurements.append(center)
    
    predict = solver.predict()
    solver.measurement(measurements)
    correct = solver.get_correct()

    for p in correct:
        idx, (x, y, _, _) = p
        frame = cv2.putText(frame, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        frame = cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 0), 2)

    print(track.object_count)
    # cv2.imshow('frame', frame)
    # key = cv2.waitKey(50) & 0xff
    # if key == 27:
    #     break
    ret, frame = video.read()
