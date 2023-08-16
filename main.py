import cv2
import os


def extract_images_from_scan(scan_path, output_dir):

    img_count = 0

    image = cv2.imread(scan_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Progowanie obrazu: wszystkie piksele, które nie są czysto białe, są ustawiane na czarno
    _, thresholded = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV) # 245 to wartość progowa; możesz ją dostosować

    # show image
    cv2.imshow("Image", thresholded)
    cv2.waitKey(0)

    # # Wyszukanie konturów
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for index, c in enumerate(contours):
        # Pobranie współrzędnych obszaru konturu
        x, y, w, h = cv2.boundingRect(c)

        # Dodatkowe sprawdzenie, czy kontur jest wystarczająco duży, by uniknąć zbyt małych fragmentów
        if w > 250 and h > 250:  # Możesz dostosować te wartości

            print(f"Kontur {index}: Szerokość = {w}, Wysokość = {h}") 
            # Eksport konturu jako obraz
            roi = image[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_dir, f'image_{index}.png'), roi)
            img_count += 1

    return img_count 


def main():
    input_dir = "Scans/Album1"
    output_dir = "Images/Album1"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ind = 0
    total_count = 0
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            count = extract_images_from_scan(os.path.join(input_dir, filename), output_dir)
            ind += 1
            print(f"Processed {filename}, {ind} out of {len(os.listdir(input_dir))}, {count} images extracted")
            total_count += count
    print (f"Total images extracted: {total_count}")

if __name__ == "__main__":
    main()
