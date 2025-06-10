from dotenv import load_dotenv
import cv2
import os
import numpy as np
import google.generativeai as genai
import base64
import re
import json
from prompt import construct_action_prompt
from annotation import annotate_image, preprocess_image
from shutil import copy2
from datetime import datetime
import serial
import time

arduino = serial.Serial(port='/dev/cu.usbmodem21101', baudrate=9600, timeout=1)

# Load environment variables
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("API_KEY"))

# Directory for raw images
RAW_IMAGES_DIR = "images"

def capture_image_from_camera(camera_index=0):
    """
    Capture an image from the specified camera and save it to the `images/` folder.
    Parameters:
    - camera_index: Index of the camera to use (1 for external camera).
    """
    camera = cv2.VideoCapture(camera_index)  # Open the specified camera
    if not camera.isOpened():
        print(f"Error: Camera with index {camera_index} not found or could not be initialized.")
        return None

    print("Press 'Enter' to capture an image.")
    image_path = None  # Ensure the variable is defined
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        # Display the camera feed
        cv2.imshow("Camera Feed", frame)

        # Wait for the 'Enter' key press to capture the image
        if cv2.waitKey(1) & 0xFF == 13:  # ASCII for Enter is 13
            image_name = "camimg.jpg"  # Saved image name
            image_path = os.path.join(RAW_IMAGES_DIR, image_name)
            cv2.imwrite(image_path, frame)
            print(f"Image captured and saved as {image_name} in {RAW_IMAGES_DIR}/")
            break

    # Release the camera and close the display window
    camera.release()
    cv2.destroyAllWindows()

    return image_path

def annotate_action_on_image(image_path, action):
    """
    Annotates the chosen action on the image in yellow text.
    Parameters:
    - image_path: Path to the image to annotate.
    - action: The action integer returned by Gemini (0 to 5).
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image for action annotation: {image_path}")
        return

    # Define annotation text and position
    annotation_text = f"Executed Action: {action}"
    position = (50, 50)  # Top-left corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 255)  # Yellow
    thickness = 2

    # Add text annotation to the image
    cv2.putText(img, annotation_text, position, font, font_scale, font_color, thickness)

    # Save the updated image
    cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Action annotation added: {annotation_text}")


def generate_response(image_path, goal, turn_around_available):
    """
    Generate a response based on the image and goal.

    Parameters:
    - image_path: Path to the input image.
    - goal: The navigation goal.
    - turn_around_available: Whether turning around is a valid option.

    Returns:
    - Full response from the model.
    """
    try:
        # Read the image as binary and encode in base64
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Generate the prompt
        num_actions = 5  # Example number of actions
        prompt = construct_action_prompt(goal, num_actions, turn_around_available)

        # Use the Gemini model to generate the response
        model = genai.GenerativeModel("gemini-1.5-flash-001")
        response = model.generate_content(
            [
                {"mime_type": "image/jpeg", "data": encoded_image},
                prompt
            ]
        )

        # Return the full response
        return response

    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def parse_action(response):
    """
    Extracts the action number from the response.

    Parameters:
    - response: The full response from the Gemini model.

    Returns:
    - action (int): The extracted action number, or -1 if parsing fails.
    """
    try:
        # Extract the text content of the response
        content = response.text

        # Use regex to find the JSON-like substring
        match = re.search(r'\{.*?\}', content)
        if match:
            action_json = match.group(0)
            action_json = action_json.replace("'", '"')  # Convert single quotes to double quotes for JSON compatibility
            action_dict = json.loads(action_json)
            return int(action_dict.get("action", -1))  # Return the action number or -1 if not found
    except Exception as e:
        print(f"Error parsing action: {e}")

    return -1  # Default action if parsing fails

def main():
    """
    Main program function.
    Continuously captures images, processes them, and sends to Gemini for responses.
    """
    try:
        while True:
            # Define the goal and other parameters
            # goal = "You are a Solar Panel Based robot, go to where you can get maximum sunlight"  # Example goal, modify as needed
            goal = "Sunlight"
            # goal = "GrayBox"
            turn_around_available = True
            print("\n--- Ready to capture a new image ---")
            
            # Capture an image from the camera
            image_path = capture_image_from_camera()
            if not image_path:
                print("No image captured. Exiting...")
                break

            # Preprocess the captured image
            temp_resized_path = preprocess_image(image_path)

            # Annotate the resized image
            annotated_output_path = "annotated_output.jpg"
            annotate_image(temp_resized_path, annotated_output_path)

            # Copy and paste image in runs folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_output_path = os.path.join("runs", f"annotated_{goal}{timestamp}.jpg")
            copy2(annotated_output_path, run_output_path)

            # Generate the response
            response = generate_response(annotated_output_path, goal, turn_around_available)

            # Parse and return the action
            if response:
                action = parse_action(response)
                print(f"\nFull Response: {response.text}")
                print(f"Extracted Action: {action}")
                arduino.write(f'{action}'.encode())
                time.sleep(1)
            else:
                print("\nFailed to generate a response.")

            # Annotate the action on the image
            annotate_action_on_image(run_output_path, action)

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting gracefully...")

if __name__ == "__main__":
    main()
