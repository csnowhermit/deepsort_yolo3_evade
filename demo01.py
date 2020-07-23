import cv2
from PIL import Image

person_boxs = [2, 3, 4, 5]    # 左上宽高

print(person_boxs[0],
      person_boxs[1],
      person_boxs[0] + person_boxs[2],
      person_boxs[1] + person_boxs[3])

person_boxs.__delitem__(2)
print(person_boxs)

