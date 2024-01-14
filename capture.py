import numpy as np
import cv2
import transform
from transform import img_path
from PIL import Image
cap = cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    # img_frame=Image.fromarray(frame)
    fin_result=img_path(frame)
    # print(type(fin_result))
    img = fin_result.convert("RGB")
    arr = np.array(img, np.uint8)
    # print(arr.shape)
    # break
    # print(type(arr))
    # cv2.imshow('frame',frame)
    arr=cv2.resize(arr, (0,0), fx=2, fy=2)
    cv2.imshow('fin_result',arr)
#     new_frame=img_path(img_frame)

#     new_frame.show()

    if cv2.waitKey(1)== ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# fin_result=img_path('Johnny-Depp-Gellert-Grindelwald-Casting.jpg')
# fin_result.show()