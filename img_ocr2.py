import re
import os
import numpy as np
import cv2
import math
import pytesseract
from PIL import Image
from scipy import ndimage
from tesserocr import PyTessBaseAPI, OEM
from unidecode import unidecode
import argparse
import datetime
import pyocr.builders

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'
TESSDATA_PREFIX = 'C:/Program Files/Tesseract-OCR'


def rotate_img(im):
    colorImage = Image.open(im)
    transposed = colorImage.transpose(Image.ROTATE_90)
    # transposed.show()
    try:
        img_str = str(im).split('.')[0]
        img_trans_name = str(img_str) + '.png'
        # nx, ny = transposed.size
        # im2 = transposed.resize((int(nx * 2.5), int(ny * 2.5)), Image.BICUBIC)
        transposed.save(img_trans_name, dpi=(520, 520))
        return img_trans_name
    except:
        file1 = open('files/ocr2.txt', 'w')
        file1.writelines('Image is not clear')
        file1.close()
        print("Image is not clear")
        exit()


def test(imPath, ro):
    if ro == 0:
        return imPath
    else:
        # print(imPath)
        try:
            im = cv2.imread(str(imPath), cv2.IMREAD_COLOR)
            newdata = pytesseract.image_to_osd(im)
            a = re.search('(?<=Rotate: )\d+', newdata).group(0)
            ro = int(a)
            # print(ro)

            if ro != 0:
                imPath = rotate_img(imPath)
            return test(imPath, ro)
        except:
            file1 = open('files/ocr2.txt', 'w')
            file1.writelines('Image is not clear')
            file1.close()
            exit()


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

dt = datetime.datetime.now()
dt = str(dt).replace("-", "")
dt = str(dt).replace(" ", "")
dt = str(dt).replace(":", "")
dt = str(dt).replace(".", "")
dt = str(dt).replace("/", "")

# print(args["image"])

img_name = args["image"]
name, ext = os.path.splitext(str(img_name))
image_name = 'temp/jfs' + str(dt) + 'a.png'
norm_img_name = 'temp/jfs' + 'aq.png'
img_rotated_name = 'temp/jfs' + 'qrotated.png'
img_rotated_name_hd = 'temp/jfs' + 'qrotated.png'
fname = str(name).split('/')[-1]
file_name = 'files/ocr2.txt'
# file_name = name + '_file.txt'


# img_name = 'images/jfs2.png'
# image_name = 'images/jfsa.png'
# norm_img_name = 'images/jfsaq.png'
# img_rotated_name = 'images/jfsqrotated.png'
# img_rotated_name_hd = 'images/jfsqrotated.png'

# print(img_name)

try:
    im = cv2.imread(str(img_name), cv2.IMREAD_COLOR)
    newdata = pytesseract.image_to_osd(im)
    # image_name = img_name
    cv2.imwrite(image_name, im)
except:
    im = Image.open(img_name)
    nx, ny = im.size
    im2 = im.resize((int(nx * 2.5), int(ny * 2.5)), Image.BICUBIC)
    im.save(image_name, dpi=(520, 520))

image = cv2.imread(image_name)
dilated_img = cv2.dilate(image[:, :, 1], np.ones((7, 7), np.uint8))
bg_img = cv2.medianBlur(dilated_img, 21)

# --- finding absolute difference to preserve edges ---
diff_img = 255 - cv2.absdiff(image[:, :, 1], bg_img)

# --- normalizing between 0 to 255 ---
norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
# cv2.imshow('norm_img', cv2.resize(norm_img, (0, 0), fx=0.5, fy=0.5))
# cv2.waitKey(0)

th = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv2.imshow('th', cv2.resize(th, (0, 0), fx=0.5, fy=0.5))
# cv2.waitKey(0)

cv2.imwrite(norm_img_name, th)
img_before = cv2.imread(norm_img_name)

img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

angles = []

for x1, y1, x2, y2 in lines[0]:
    cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 3)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angles.append(angle)

median_angle = np.median(angles)
img_rotated = ndimage.rotate(img_before, median_angle)

# print("Angle is {}".format(median_angle))
cv2.imwrite(img_rotated_name, img_rotated)

img_rotated_name1 = test(img_rotated_name_hd, 1)
# print(img_rotated_name1)
# print("--------------------------------------------------- 1 PyAPI --------------------------------------")

#file1 = open(file_name, 'w')
#file1.writelines("--------------------------------------------------- 1 PyAPI --------------------------------------\n")

with PyTessBaseAPI(lang="best/eng", oem=OEM.LSTM_ONLY) as api:
    # image = 'images/jfs1.png'
    api.SetImageFile(img_rotated_name1)
    # print(api.GetUTF8Text())
    file1 = open(file_name, 'w')
    file1.write(api.GetUTF8Text())

file1.close()

tools = pyocr.get_available_tools()
tool = tools[0]
line_and_word_boxes = tool.image_to_string(
    Image.open(img_rotated_name1),
    lang='best/eng',
    builder=pyocr.builders.TextBuilder()
)
# print("--------------------------------------------------- 2 PyOCR --------------------------------------")
# print(line_and_word_boxes)

original = pytesseract.image_to_string(img_rotated_name1)
# print("-------------------------------------------------- 3 Pytess --------------------------------------")
# print(original)

original1 = pytesseract.image_to_string(img_rotated_name1, lang='eng', config='--psm 6')
# print("-------------------------------------------------- 4 Pytess --------------------------------------")
# print(original1)

original2 = pytesseract.image_to_string(img_rotated_name1, lang='best/eng', config='--psm 6')
# print("-------------------------------------------------- 5 Pytess --------------------------------------")
# print(original2)

bb="Task completed Succesfully"
with open(file_name, "ab") as f:
    # f.seek(40)
    b = bytearray(b"\n")
    f.write(b)
    f.write(line_and_word_boxes.encode("UTF-8"))
    b1 = bytearray(
        b"\n")
    f.write(b1)
    f.write(original.encode("UTF-8"))
    b2 = bytearray(b"\n")
    f.write(b2)
    f.write(original1.encode("UTF-8"))
    b3 = bytearray(b"\n")
    f.write(b3)
    f.write(original2.encode("UTF-8"))
    b4 = bytearray(b"\n")
    f.write(b4)
    f.write(bb.encode("UTF-8"))

# try:
#    os.chmod(image_name, 0o777)
#    os.remove(image_name)
#    os.chmod(img_rotated_name1, 0o777)
#    os.remove(img_rotated_name1)
#    os.chmod(norm_img_name, 0o777)
#    os.remove(norm_img_name)
#    os.chmod(img_rotated_name, 0o777)
#    os.remove(img_rotated_name)
# except Exception as ef:
#    print(str(ef))
