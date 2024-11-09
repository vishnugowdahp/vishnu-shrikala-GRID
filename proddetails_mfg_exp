import cv2
from pyzbar.pyzbar import decode
import requests
import pytesseract
import re
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import time
import hashlib
from datetime import datetime

# Set the path for Tesseract executable (Update the path according to your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Dictionary to store item details and their counts
item_db = {}

def extract_barcode_from_image(image_path):
    """Extract barcode from the given image path."""
    img = cv2.imread(image_path)
    decoded_objects = decode(img)

    if decoded_objects:
        barcode = decoded_objects[0].data.decode('utf-8')
        print(f"Extracted Barcode: {barcode}")
        return barcode
    else:
        print("No barcode found.")
        return None

def get_product_details_from_open_food_facts(barcode, retries=3, delay=2):
    """Fetch product details from Open Food Facts API."""
    url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            product_info = response.json()
            
            if product_info['status'] == 1:
                return product_info['product']
            else:
                print("Product not found in the database.")
                return None
            
        except requests.exceptions.Timeout:
            print(f"Attempt {attempt + 1} of {retries}: Request timed out. Retrying in {delay} seconds...")
            time.sleep(delay)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break  # Exit the loop on other request errors

    return None

def remove_unwanted_spaces(input_text):
    """Remove unwanted spaces from the input text."""
    cleaned_text = ' '.join(input_text.split())
    return cleaned_text

def preprocess_image(image_path, contrast=1.5, brightness=0, threshold_value=128):
    """Preprocess the image for OCR."""
    image = cv2.imread(image_path)

    # Resize the image to improve OCR accuracy
    height, width = image.shape[:2]
    new_size = (width * 2, height * 2)
    image = cv2.resize(image, new_size)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply contrast and brightness adjustments
    adjusted = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)

    # Use CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(adjusted)

    # Apply a binary threshold
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    return gray, thresh

# Regular expression pattern to match date formats
date_pattern = r'(\d{1,2}\/\d{1,2}\/\d{2,4})|(\d{1,2}\/\d{4})|(\d{1,2}\/\d{2})|(\d{1,2}\/[A-Za-z]{3}\/\d{4})|(\d{1,2}\.\d{1,2}\.\d{2,4})|([A-Za-z]{3,9},\s*\d{4})|([A-Za-z]{3,9},\s*\d{1,2})|(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})|(\d{1,2}[A-Za-z]{3,9}\d{4})|(\d{1,2}\s+[A-Za-z]{3,9})|(\d{1,2}-\d{1,2}-\d{2,4})|(\d{1,2}\/\d{2})|([A-Za-z]{3}\/\d{2})'

def extract_dates(input_text):
    matches = re.findall(date_pattern, input_text)
    extracted_dates = []

    for match in matches:
        for date in match:
            if date:  # Only add non-empty matches
                extracted_dates.append(date)

    return extracted_dates

def convert_to_date(date_str):
    """Convert string to datetime object based on various formats."""
    for fmt in ("%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d.%m.%Y", "%B, %Y", "%b, %Y", "%d %B %Y", "%d %b %Y", "%b%y", "%b %y", "%d%b%Y", "%d%b%y", "%m/%y", "%b/%y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def calculate_image_hash(image_path):
    """Calculate and return the hash of the image for identification."""
    with open(image_path, 'rb') as f:
        file_data = f.read()
        return hashlib.md5(file_data).hexdigest()

def track_image_upload(image_hash):
    """Track how many times an image has been uploaded based on its hash."""
    if image_hash in item_db:
        item_db[image_hash]['count'] += 1
    else:
        item_db[image_hash] = {'count': 1, 'dates': None}

    return item_db[image_hash]['count']

def try_all_parameter_combinations(image_path, image_hash):
    """Try all combinations of contrast, brightness, and threshold until MFG and EXP dates are extracted."""
    contrast_range = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    brightness_range = [-1, 0, 1]
    threshold_range = [123, 124, 125, 126, 127, 128, 129, 130]

    for contrast in contrast_range:
        for brightness in brightness_range:
            for threshold_value in threshold_range:
                gray_image, processed_image = preprocess_image(image_path, contrast, brightness, threshold_value)
                extracted_text = pytesseract.image_to_string(processed_image)

                # Clean and extract dates from the text
                cleaned_text = remove_unwanted_spaces(extracted_text)
                extracted_dates = extract_dates(cleaned_text)

                # Convert the extracted dates into datetime objects
                date_objects = [convert_to_date(date) for date in extracted_dates]
                date_objects = [date for date in date_objects if date]  # Remove None values

                if len(date_objects) >= 2:
                    date_objects.sort()  # Sort to identify MFG and EXP dates
                    mfg_date = date_objects[0]
                    exp_date = date_objects[1]

                    # Store the dates in the item database
                    item_db[image_hash]['dates'] = {
                        'mfg_date': mfg_date.strftime("%d/%m/%Y"),
                        'exp_date': exp_date.strftime("%d/%m/%Y")
                    }

                    return  # Stop once we find both dates

    print("Unable to extract both MFG and EXP dates with the given combinations.")

def select_image():
    """Open a file dialog to select an image."""
    Tk().withdraw()  # Close the root window
    image_path = askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    return image_path

def display_stored_data():
    """Display the stored data so far."""
    print("----- Stored Data -----")
    for image_hash, data in item_db.items():
        print(f"ID: {image_hash} | Count: {data['count']} | MFG Date: {data['dates']['mfg_date'] if data['dates'] else 'N/A'} | EXP Date: {data['dates']['exp_date'] if data['dates'] else 'N/A'}")
    print("-----------------------")

def main():
    """Main function to run the barcode extraction and retrieve product details."""
    while True:
        image_path = select_image()
        if not image_path:
            print("No image selected.")
            return

        # Extract barcode from the image
        barcode = extract_barcode_from_image(image_path)
        if barcode:
            print("Fetching product details...")
            product_details = get_product_details_from_open_food_facts(barcode)

            if product_details:
                print("----- Product Details -----")
                print(f"Product Name: {product_details.get('product_name', 'N/A')}")
                print(f"Brand: {product_details.get('brands', 'N/A')}")
                print(f"Description: {product_details.get('description', 'N/A')}")
                print(f"Image URL: {product_details.get('image_url', 'N/A')}")
                print(f"Nutritional Info: {product_details.get('nutriments', 'N/A')}")
                print("---------------------------")

                # Calculate the image hash
                image_hash = calculate_image_hash(image_path)

                # Track image upload count
                count = track_image_upload(image_hash)

                # Try to extract MFG and EXP dates from the image
                try_all_parameter_combinations(image_path, image_hash)

                # Display MFG and EXP dates if available
                current_data = item_db[image_hash]
                if current_data['dates']:
                    print(f"MFG Date: {current_data['dates']['mfg_date']}")
                    print(f"EXP Date: {current_data['dates']['exp_date']}")
                    print(f"ID: {image_hash}")  # Using barcode as ID
                else:
                    print("MFG and EXP dates could not be extracted.")

                # Display stored data so far
                display_stored_data()

            else:
                print("Failed to retrieve product details.")

        # Ask user if they want to continue
        continue_choice = input("Do you want to scan another image? (yes/no): ").strip().lower()
        if continue_choice not in ['yes', 'y']:
            print("Exiting the program.")
            break

if __name__ == "__main__":
    main()
