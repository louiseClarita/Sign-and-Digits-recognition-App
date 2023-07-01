import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import ImageDraw, ImageFont
from PIL import Image, ImageDraw, ImageFont, ImageGrab


def center_crop(frame):
    # Perform center cropping on the frame
    height, width, _ = frame.shape
    start = abs(height - width) // 2
    if height > width:
        cropped_frame = frame[start: start + width]
    else:
        cropped_frame = frame[:, start: start + height]
    return cropped_frame


def main():
    # Constants and setup
    index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')
    mean = 0.485 * 255.
    std = 0.229 * 255.

    # Load the exported ONNX model
    ort_session = ort.InferenceSession("signlanguage.onnx")

    # Open webcam
    capture = cv2.VideoCapture(1)

    while True:
        # Capture frame from webcam
        ret, frame = capture.read()

        # preprocess data
        frame = center_crop(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        x = cv2.resize(frame, (28, 28))
        x = (x - mean) / std

        x = x.reshape(1, 1, 28, 28).astype(np.float32)
        y = ort_session.run(None, {'input': x})[0]

        index = np.argmax(y, axis=1)
        letter = index_to_letter[int(index)]

        cv2.putText(frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.imshow("Sign Language Translator", frame)

        # Wait for key press and perform actions
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('c') or key == ord('s'):
            action = "image captured" if key == ord('c') else "image captured and quitting"
            print(action)
            cv2.imwrite('captured_image.jpg', frame)
            image = Image.open('captured_image.jpg')
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()  # Replace with the path to your desired font
            #draw.text((10, 10), str(letter), fill=(0, 0, 0), font=font)
            image.show()
            if key == ord('s'):
                break

    # Release resources
    capture.release()
    cv2.destroyAllWindows()
main()
if __name__ == '__main__':
    print("please make sure you have a clear background, the data used to train this model is specific to clear backgrounds, we will make sure to find more generic data later on!")
    print("Press Q to exit")

    #main()
