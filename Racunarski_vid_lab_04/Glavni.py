import numpy as np
import cv2 as cv
import imutils
import time
import matplotlib.pyplot as plt

RESOURCE_PATH = 'resources/'
OUTPUT_PATH = 'results/'
STEP_SIZE = 180
MIN_CONFIDENCE_CAT = 0.81
MIN_CONFIDENCE_DOG = 0.55
SCALES = [1, 2, 4]  # Scales for multi-scale detection


def detect_objects(img, net, classes):
    img_output = img.copy()

    for scale in SCALES:
        size = int(img.shape[1] / scale)
        scaled_img = imutils.resize(img, width=size, height=size)
        scaled_step = STEP_SIZE

        for y in range(0, scaled_img.shape[0] - STEP_SIZE + 1, scaled_step):
            for x in range(0, scaled_img.shape[1] - STEP_SIZE + 1, scaled_step):

                scaled_x = int(x)
                scaled_y = int(y)

                # Ensure we don't exceed bounds
                if scaled_y + STEP_SIZE > scaled_img.shape[0] or scaled_x + STEP_SIZE > scaled_img.shape[1]:
                    continue

                # Extract the region from the scaled image
                part_img = scaled_img[scaled_y:scaled_y + STEP_SIZE, scaled_x:scaled_x + STEP_SIZE]

                # Preprocess the image for GoogLeNet
                blob = cv.dnn.blobFromImage(part_img, 1, (224, 224), (104, 117, 123))
                net.setInput(blob)
                preds = net.forward()

                # Get the top prediction
                idx = np.argmax(preds[0])
                confidence = preds[0][idx]
                label = classes[idx]

                # Map detection to the original image scale
                x_original = int(scaled_x * scale)
                y_original = int(scaled_y * scale)
                width_original = int(STEP_SIZE * scale)
                height_original = int(STEP_SIZE * scale)

                # Check detection and draw rectangles
                if confidence >= MIN_CONFIDENCE_CAT and 'cat' in label:
                    cv.rectangle(img_output, (x_original, y_original),
                                 (x_original + width_original, y_original + height_original), (0, 0, 255), 2)
                    cv.putText(img_output, "CAT", (x_original + 5, y_original + 25),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                elif confidence >= MIN_CONFIDENCE_DOG and 'dog' in label:
                    cv.rectangle(img_output, (x_original, y_original),
                                 (x_original + width_original, y_original + height_original), (0, 255, 255), 2)
                    cv.putText(img_output, "DOG", (x_original + 5, y_original + 25),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return img_output


def main():

    img = cv.imread(f'{RESOURCE_PATH}input.png')
    if img is None:
        print("Failed to load image.")
        return

    # Crop to the white-bordered region
    img = img[293:1013, 244:1684]

    # Load class labels and pre-trained model
    rows = open(f'{RESOURCE_PATH}synset_words.txt').read().strip().split("\n")
    classes = [r[r.find(" ") + 1:].split(",")[0].lower() for r in rows]
    net = cv.dnn.readNetFromCaffe(
        f'{RESOURCE_PATH}bvlc_googlenet.prototxt', f'{RESOURCE_PATH}bvlc_googlenet.caffemodel')

    # Detect objects with multi-scale sliding windows
    img_detected = detect_objects(img, net, classes)

    # Save and display output image
    cv.imshow('Output', img_detected)
    cv.imwrite(f'{OUTPUT_PATH}output.jpg', img_detected)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f'Execution time: {end_time - start_time}')