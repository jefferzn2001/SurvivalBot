import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Resize the image to a 16:10 aspect ratio and save it as a temporary file.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    target_width = 1200  # Fit for 16:10 screens
    target_height = 750
    img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    temp_path = 'temp_resized.jpg'
    cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])  # Save with high quality
    return temp_path

def annotate_image(image_path, output_path):
    """
    Annotate the image with lines originating from the bottom center
    and labeled with circular markers.
    """
    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    height, width = img.shape[:2]

    # Define the true bottom center of the image
    base_x = width // 2
    bottom_y = height - 1  # Connect arrows exactly to the bottom boundary

    # Line origins (slightly offset for better separation)
    offsets = [-100, -50, 0, 50, 100]
    origins = [(base_x + offset, bottom_y) for offset in offsets]

    # Directions, lengths, and labels
    angles = [-45, -20, 0, 20, 45]
    lengths = [200, 220, 240, 220, 200]  # Line lengths
    labels = ['1', '2', '3', '4', '5']

    # Draw lines, labels, and circular markers
    for i, angle in enumerate(angles):
        origin_x, origin_y = origins[i]
        length = lengths[i]
        angle_rad = np.deg2rad(angle)
        end_x = int(origin_x + length * np.sin(angle_rad))
        end_y = int(origin_y - length * np.cos(angle_rad))

        # Ensure the line ends seamlessly at the circular marker
        end_y = max(end_y, bottom_y - length)

        # Draw the line (red)
        cv2.line(img, (origin_x, origin_y), (end_x, end_y), (0, 0, 255), 2)

        # Draw the circular marker
        cv2.circle(img, (end_x, end_y), 15, (0, 0, 0), -1)  # Black outer circle
        cv2.circle(img, (end_x, end_y), 12, (255, 255, 255), -1)  # White inner circle

        # Label the marker
        cv2.putText(img, labels[i], (end_x - 7, end_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Save the annotated image
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

if __name__ == "__main__":
    input_path = "camimg.jpg"  # Replace with your input image file path
    output_path = "annotated_output.jpg"  # Replace with your desired output file name

    # Preprocess the image (resize and save temp file)
    resized_path = preprocess_image(input_path)

    # Annotate the resized image
    annotate_image(resized_path, output_path)

    # Display the annotated image
    annotated_img = cv2.imread(output_path)
    cv2.imshow("Annotated Image", annotated_img)
    print("Press any key to close the image window...")
    key = cv2.waitKey(0)
    if key != -1:
        cv2.destroyAllWindows()
