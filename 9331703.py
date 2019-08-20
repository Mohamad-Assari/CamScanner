import cv2
import numpy 

locationOfPic=input("please give the location of the file\n")

inputImage=cv2.imread(locationOfPic)

inputImage=cv2.resize(inputImage, (800, 800))

grayInput=cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

blurredInput = cv2.GaussianBlur(grayInput, (1,1), 0)


def threshold(blur):
	edged = cv2.Canny(blur, 35, 45)
	c,h = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	sortedContors = sorted(c, key=cv2.contourArea, reverse=True)
	for i in sortedContors:
		p = cv2.arcLength(i, True)
		pointz = cv2.approxPolyDP(i, 0.1*p, True)

		if len(pointz) == 4:
			target = pointz
			break
	pointz = crop(target)
	return pointz

def crop(corner):
    corner = corner.reshape((4, 2))
    cornersResize = numpy.zeros((4, 2), dtype=numpy.float32)
    add = corner.sum(1)
    cornersResize[0] = corner[numpy.argmin(add)]
    cornersResize[2] = corner[numpy.argmax(add)]
    sth = numpy.diff(corner, axis=1)
    cornersResize[1] = corner[numpy.argmin(sth)]
    cornersResize[3] = corner[numpy.argmax(sth)]
    return cornersResize


points = threshold(blurredInput)
size = abs(points[0][0] - points[1][0])

def transforrm(original_image, endpoints):
    pts = numpy.float32([[0, 0], [size, 0], [size, size], [0, size]])
    perspective = cv2.getPerspectiveTransform(endpoints, pts)
    perspective_image = cv2.warpPerspective(original_image, perspective, (size, size))
    return perspective_image

perspective_image = transforrm(inputImage, points)

img = perspective_image
lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(1,1))
cl = clahe.apply(l)
limg = cv2.merge((cl,a,b))
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

kernel = numpy.array([[-1, -1, -1],
                              [-1, 9.5, -1],
                              [-1, -1, -1]])
final = cv2.filter2D(final, -2, kernel)
final = cv2.resize(final, (size, size))
cv2.imshow('output', final)
cv2.imwrite('output.jpg', final)
cv2.waitKey(0)
