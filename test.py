import numpy as np
import cv2
from PIL import Image
from mss import mss
from time import time
import pytesseract

sct = mss()
win1 = {"top": 280, "left": 425, "width": 48, "height": 25}
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
while 1:
    screenshot = sct.grab(win1)
    img = np.array(screenshot)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow("test", img_rgb)
    text = pytesseract.image_to_string(img_rgb)
    print(text)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
