import cv2
import os


def extract_images_from_scan(scan_path, output_dir):

    img_count = 0

    image = cv2.imread(scan_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    _, thresholded = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

    # # Wyszukanie konturów
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for index, c in enumerate(contours):
        # Pobranie współrzędnych obszaru konturu
        x, y, w, h = cv2.boundingRect(c)

        if w > 250 and h > 250 and w < 2000 and h < 2000:
            # print(f"Kontur {index}: Szerokość = {w}, Wysokość = {h}") 
            roi = image[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_dir, f'image_{index}.png'), roi)
            img_count += 1

    if img_count < 3:
        # show input image
        cv2.imshow("Input", image)
        cv2.waitKey(0)
    
    return img_count


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
    total_count = 0
    no_input_imgs = len(os.listdir(input_dir))

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            count = extract_images_from_scan(os.path.join(input_dir, filename), output_dir)
            ind += 1
            print(f"Processed {ind} out of {no_input_imgs}, {filename} ")
            total_count += count

    print (f"Total images extracted: {total_count}")


if __name__ == "__main__":
    main()