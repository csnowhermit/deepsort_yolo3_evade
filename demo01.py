import cv2

list1 = [('child', 0.4166157, [542.0016, 278.24387, 600.6658, 355.25095])]
list2 = [('head', 0.61927605, [537.6127, 283.58765, 604.105, 350.898]),
         ('head', 0.9575744, [546.5691, 78.46593, 627.095, 171.85654])]

# print(set(list1).union(set(list2)))
list3 = list1 + list2

for ls in (list1 + list2):
    print(ls)

child_classes = ['child']
child_scores = [0.4166157]
child_boxs = [[542.0016, 278.24387, 600.6658, 355.25095]]

adult_classes = ['head', 'head']
adult_scores = [0.61927605, 0.9575744]
adult_boxs = [[537.6127, 283.58765, 604.105, 350.898], [546.5691, 78.46593, 627.095, 171.85654]]

print(child_classes + adult_classes)
print(child_scores + adult_scores)
print(child_boxs + adult_boxs)

from common.config import child_types

if 'child' in child_types:
    print("True")
