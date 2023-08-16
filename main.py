import cv2
import os
import numpy as np
import math


def extract_images_from_scan(scan_path, output_dir, total_count):
    
    img_count = 0
    img_extraction_error = False

    image = cv2.imread(scan_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)


    # Wykrywanie krawędzi za pomocą Canny
    edges = cv2.Canny(thresholded, 50, 150, apertureSize=3)

    # Wykrywanie linii za pomocą transformacji Hougha
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    # Sprawdzanie czy są jakieś linie do przetworzenia
    if lines is not None:
        rotation_angles = []

        for rho, theta in lines[:, 0]:
            # Obliczanie kąta obrotu na podstawie linii
            a = np.cos(theta)
            b = np.sin(theta)
            if b:
                angle = math.degrees(math.atan(-a/b))
                if -30 <= angle <= 30:  # Jeśli kąt jest między -30 a 30 stopniami
                    rotation_angles.append(angle)

        # Obliczanie średniego kąta obrotu i obracanie obrazu
        if rotation_angles:
            avg_angle = np.mean(rotation_angles)
            center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
            image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)


    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for index, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if w > 250 and h > 250 and w < 2000 and h < 2000:
            roi = image[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_dir, f'image_{total_count}.png'), roi)
            total_count += 1
            img_count += 1

    if img_count < 3:
        # cv2.imshow("Input", image)
        # cv2.waitKey(0)
        img_extraction_error = True

    return img_count, img_extraction_error



def main():
    input_dir = "Scans/Album1"
    output_dir = "Images/Album1"

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
    no_errors = 0
    total_count = 0
    no_input_imgs = len(os.listdir(input_dir))

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            count, img_extraction_error = extract_images_from_scan(os.path.join(input_dir, filename), output_dir, total_count)
            ind += 1
            print(f"Processed {ind} out of {no_input_imgs}, {filename} ")
            total_count += count
            if img_extraction_error:
                no_errors += 1

    print (f"Total images extracted: {total_count}, expected: {no_input_imgs * 3}")
    print (f"Total errors: {no_errors}")

if __name__ == "__main__":
    main()