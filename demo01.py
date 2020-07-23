import cv2
from PIL import Image

cap = cv2.VideoCapture(0)

# savefile = "D:/monitor_images/10.6.8.181/normal_images/10.6.8.181_2020-07-23_09:47:47.png"
savefile = "D:/monitor_images/10_6_8_181/normal_images/1.jpg"

ret, frame = cap.read()
# print(frame)

print(cv2.imwrite(savefile, frame))
print("===")

# image = Image.fromarray(frame)
# image.show()
# image.save(savefile, quality=)

# cv.saveImage(savefile, frame)

