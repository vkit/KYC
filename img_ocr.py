import cv2
import pytesseract
from PIL import Image
import sys
import os
import pyocr
import pyocr.builders
import datetime
import argparse
import unidecode
import codecs

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
fname = str(name).split('/')[-1]
image_name = 'temp/jfs' + str(dt) + '.png'
file_name = 'files/ocr.txt'

#file1 = open(file_name, 'w')

tools = pyocr.get_available_tools()
tool = tools[0]

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'
TESSDATA_PREFIX = 'C:/Program Files/Tesseract-OCR'

try:
    im = cv2.imread(str(img_name), 0)
    newdata = pytesseract.image_to_osd(im)
    #image_name = img_name
    cv2.imwrite(image_name, im)
except:
    im = Image.open(img_name)
    nx, ny = im.size
    im2 = im.resize((int(nx * 2.5), int(ny * 2.5)), Image.BICUBIC)
    im2.save(image_name, dpi=(600, 600))
    im.close()

line_and_word_boxes = tool.image_to_string(
    Image.open(image_name),
    lang='best/eng',
    builder=pyocr.builders.TextBuilder()
)
print(line_and_word_boxes)
#file1.writelines(unidecode(line_and_word_boxes))
with open(file_name, "wb") as f:
   f.write(line_and_word_boxes.encode("UTF-8"))

#file1.close()

#try:
#    os.chmod(image_name, 0o777)
#    os.remove(image_name)
#except Exception as ef:
#    print(str(ef))
