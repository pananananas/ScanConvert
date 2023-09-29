# # # # # # # # # # # # # # # # # # # # # # # # # # 
# Initial parameters

min_image_width  = 700
min_image_height = 700
max_image_width  = 3000
max_image_height = 3000

number_of_images_per_scan = 3

# Inpt and output folders (relative to main.py)
input_dir = "Scans/Album1"
output_dir = "Images/Album1"

# # # # # # # # # # # # # # # # # # # # # # # # # # 

import cv2
import os


def rotate_and_crop_image(image, angle):
    h, w = image.shape[:2]
    img_center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(img_center, angle, 1.0)

    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        rotated_and_cropped = rotated[y:y+h, x:x+w]
        return rotated_and_cropped
    else:
        return rotated


def extract_images_from_scan(scan_path, total_count):
    
    img_count = 0
    img_extraction_error = False

    image = cv2.imread(scan_path)
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    contours, _    = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for index, c in enumerate(contours):

        x, y, w, h = cv2.boundingRect(c)
        if w > min_image_width and h > min_image_height and w < max_image_width and h < max_image_height:
            _, _, angle = cv2.minAreaRect(c)

            if angle > 45:
                angle -= 90
            roi = image[y:y+h, x:x+w]
            
            rotated_roi = rotate_and_crop_image(roi, angle)

            cv2.imwrite(os.path.join(output_dir, f'image_{total_count}.png'), rotated_roi)
            total_count += 1
            img_count += 1

    if img_count != number_of_images_per_scan:
        print(f"Error extracting images from {scan_path} - found {img_count} images")
        cv2.imshow("Input", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        img_extraction_error = True

    return img_count, img_extraction_error


def main():

    if os.path.exists(output_dir):                                          # delete all contents of output_dir if it exists
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(output_dir)

    ind = 0
    num_of_errors = 0
    total_count = 0
    num_of_input_imgs = len(os.listdir(input_dir))

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".png"):
            count, img_extraction_error = extract_images_from_scan(os.path.join(input_dir, filename), total_count)
            ind += 1
            print(f"Processed {ind} out of {num_of_input_imgs}, {filename} ")
            total_count += count
            if img_extraction_error:
                num_of_errors += 1
                
    num_of_output_imgs = len(os.listdir(output_dir))
    print (f"Total images extracted: {num_of_output_imgs}, expected: {num_of_input_imgs * number_of_images_per_scan}")
    print (f"Total errors: {num_of_errors}")


if __name__ == "__main__":
    main()