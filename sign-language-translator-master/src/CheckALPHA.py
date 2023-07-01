import time

import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont
from PIL import Image, ImageDraw, ImageFont, ImageGrab



image = cv2.imread('Letter_k.jpg')
greyImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image = cv2.resize(image, (28, 28))

# Convert the image to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')
mean = 0.485 * 255.
std = 0.229 * 255.

# Load the exported ONNX model
ort_session = ort.InferenceSession("signlanguage.onnx")
x = cv2.resize(image, (28, 28))
x = (x - mean) / std

x = x.reshape(1, 1, 28, 28).astype(np.float32)
y = ort_session.run(None, {'input': x})[0]

index = np.argmax(y, axis=1)
letter = index_to_letter[int(index)]
cv2.putText(greyImage, letter, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), thickness=3)
print(letter)
#cv2.imshow("Sign checker - Letters", image)
plt.imshow(greyImage, cmap='gray')
plt.axis('off')  # Remove axis ticks and labels
plt.show()
