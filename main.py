import cv2
import os



def rotate_and_crop_image(image, angle):
    # Oblicz środek obrazu
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # Utwórz macierz obrotu
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Obróć obraz
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # Przytnij białe marginesy
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)  # znajdź największy kontur
        x, y, w, h = cv2.boundingRect(c)
        rotated_and_cropped = rotated[y:y+h, x:x+w]
        return rotated_and_cropped
    else:
        return rotated


def extract_images_from_scan(scan_path, output_dir, total_count):
    
    img_count = 0
    img_extraction_error = False

    image = cv2.imread(scan_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for index, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if w > 250 and h > 250 and w < 2000 and h < 2000:
            # Oblicz kąt obrotu dla konturu
            _, _, angle = cv2.minAreaRect(c)

            # Jeśli kąt jest większy niż 45 stopni, przekształć go
            # (zakładając, że chcemy mały kąt obrotu)
            if angle > 45:
                angle -= 90

            # Wycięcie regionu zainteresowania
            roi = image[y:y+h, x:x+w]

            # Obróć wycięty obraz
            rotated_roi = rotate_and_crop_image(roi, angle)

            cv2.imwrite(os.path.join(output_dir, f'image_{total_count}.png'), rotated_roi)
            total_count += 1
            img_count += 1

    if img_count < 3:
        print(f"Error extracting images from {scan_path} - found {img_count} images")
        cv2.imshow("Input", image)
        cv2.waitKey(0)
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