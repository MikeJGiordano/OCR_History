#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 08:35:53 2023

@author: michaelgiordano
"""

import io
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import vision
import json
import boto3
from IPython.display import display
from PIL import Image, ImageDraw
from textractor import Textractor
from textractor.visualizers.entitylist import EntityList
from textractor.data.constants import TextractFeatures, Direction, DirectionalFinderType
from tqdm import tqdm


def process_content(filename,
                    input_folder,
                    output_folder,
                    show_image, 
                    use_google_vision, 
                    use_textract, 
                    verbose):
    input_file = os.path.join(input_folder, filename)
    try:
        if use_google_vision:
            if show_image:
                print("Google Vision Output:")
            else:
                pass
            image = cv2.imread(input_file)
            # Replace the file extension with ".json"
            gcloud_json = os.path.join(output_folder, os.path.splitext(filename)[0] + "_GCloud.json")
            gcloud_boxes(image, gcloud_json, show_image, save_text_to_txt=True)
        else:
            pass
    except:
        print("Error with Cloud Vision")

    try:
        if use_textract:
            print("Running through Textract since use_textract=True")
            if show_image:
                print("Textract Output:")
            else:
                pass
            amazon_json = os.path.join(output_folder, os.path.splitext(filename)[0] + "_Textract.json")
            textract_process_image(input_file, amazon_json, show_image)
        else:
           pass
    except:
        print("Error with Textract")
        
    if verbose:
        print("Setting all parameters=True gives a basic visualization of the outputs of both Cloud Vision, defaulted as the first image, and Textract, the second image. The .txt and .json outputs for both Cloud Vision and Textract are saved in the output_folder. By setting a parameter=False, you can skip that function. For example, if use_textract=False and use_google_vision=True, this will not send the image through Textract, but will send the image through Google Vision.")
    
        

def batch_ocr(input_folder, output_folder, use_google_vision, use_textract):
    # Get the list of image files with valid extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    image_files = [filename for filename in os.listdir(input_folder) if any(filename.lower().endswith(ext) for ext in valid_extensions)]

    # Create a tqdm progress bar
    progress_bar = tqdm(total=len(image_files), desc='Processing Images', unit='image')

    for filename in image_files:
        file_path = os.path.join(input_folder, filename)

        process_content(filename, 
                        input_folder, 
                        output_folder, 
                        show_image=False, 
                        use_google_vision=use_google_vision, 
                        use_textract=use_textract, 
                        verbose=False)

        # Update the progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    print(f"All images OCR'd. text and JSON files are in folder {output_folder}")


def textract_process_image(image_path, json_output, show_image):
    # Create a Textract Client
    textract = boto3.client('textract')

    # Read the image as bytes
    with open(image_path, 'rb') as document:
        image_bytes = document.read()

    # Call Textract DetectDocumentText to analyze the document
    response = textract.detect_document_text(Document={'Bytes': image_bytes})

    # Save the Textract JSON output to a specified location
    with open(json_output, 'w') as json_output_file:
        json.dump(response, json_output_file, indent=4)

    # Save the detected text to a .txt file
    with open(json_output.replace('.json', '.txt'), 'w') as txt_file:
        for item in response['Blocks']:
            if item['BlockType'] == 'WORD':
                txt_file.write(item['Text'] + '\n')

    # Get the detected text blocks
    blocks = response['Blocks']

    # Open the image using PIL
    image = Image.open(io.BytesIO(image_bytes))

    # Check if the image is grayscale and convert to color if needed
    if image.mode == 'L':
        image = image.convert('RGB')

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Iterate through the blocks to draw bounding boxes in blue
    for block in blocks:
        if block['BlockType'] == 'WORD':
            polygon = block['Geometry']['Polygon']
            points = [(p['X'] * image.width, p['Y'] * image.height) for p in polygon]
            draw.polygon(points, outline='blue')

    if show_image:
        # Display the image with bounding boxes
        plt.figure(figsize=(10, 20))
        plt.imshow(image)
        plt.show()


#Set default dictionary
default = {
    "left_margin_percent":15,
    "top_margin_percent":15,
    "vsplit_percent":0,
    "hsplit_percent":0,
    "brightness_factor" : 1,
    "contrast_factor" : 1
}




######
# Extract Tables
######


def extract_table(extractor, filename, input_folder, output_folder):
    
    input_file = input_file = os.path.join(input_folder, filename)
    image = Image.open(input_file)
    # Analyze the document and specify you want to extract tables
    document = extractor.analyze_document(
        file_source=image,
        features=[TextractFeatures.TABLES],
        save_image=True
        )

    #Show the summary statistics of the detected objects

    document

    #Create a variable for the detected words
    words_entity_list = document.words
    #Create a variable for the detected tables
    table_list = document.tables

    #Load the document image with cv2. This is necessary to draw the bounding boxes
    sing_image = cv2.imread(input_file)
    #Draw the bounding boxes in Textract
    textract_boxes(sing_image, words_entity_list, table_list, show_image=True)
    
    batch_run = input("Do you want to batch extract the Excel tables from images in the input_folder? (y/n): ")
    if batch_run.lower() =="y":
        print("The .xlsx output files will be saved in your output_folder")
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
        image_files = [filename for filename in os.listdir(input_folder) if any(filename.lower().endswith(ext) for ext in valid_extensions)]

        # Create a tqdm progress bar
        progress_bar = tqdm(total=len(image_files), desc='Processing Images', unit='image')

        for filename in image_files:
             input_file = input_file = os.path.join(input_folder, filename)
             image = Image.open(input_file)
     
             document = extractor.analyze_document(
                file_source=image,
                features=[TextractFeatures.TABLES],
                save_image=False
                )
             file_name, current_extension = os.path.splitext(filename)
             new_filename = file_name + '.xlsx' 
             output_path = os.path.join(output_folder, new_filename)
             document.tables[0].to_excel(output_path)
     
     
             # Update the progress bar
             progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()




######
#Preprocess only
######

def preprocess_image(filename, input_folder, output_folder, left_margin_percent, top_margin_percent, vsplit_percent, hsplit_percent, brightness_factor, contrast_factor):
    # Get the height and width of the input image
    image = cv2.imread(os.path.join(input_folder, filename))
    height, width = image.shape[0:2]
    
    split_question = input('Do you want to split this image into two separate images? (y/n):')
    if split_question.lower() =='n':
        vsplit_percent = 0
        hsplit_percent = 0
        pass
    if split_question.lower() == 'y':
        vert_horiz = input('Do you want to split it Vertically or Horizontally? (v/h)')
        if vert_horiz.lower() == 'v' and vsplit_percent ==0:
            default['vsplit_percent'] = 50
            vsplit_percent = 50
            hsplit_percent = 0
        if vert_horiz.lower() == 'v' and vsplit_percent !=0:
            hsplit_percent = 0            
        if vert_horiz.lower() == 'h' and hsplit_percent ==0:            
            vsplit_percent = 0
            hsplit_percent = 50
            default['hsplit_percent'] = 50
        if vert_horiz.lower() == 'h' and hsplit_percent !=0:
            vsplit_percent = 0    
        else:
            pass

    # Calculate margin values based on percentages
    LeftMargin = int(width * (left_margin_percent / 100))
    TopMargin = int(height * (top_margin_percent / 100))
    RightMargin = int(width - LeftMargin)
    BottomMargin = int(height - TopMargin)
    
    # Calculate split lines based on percentages
    VSplit = int(width * (vsplit_percent / 100))
    HSplit = int(height * (hsplit_percent / 100))

    # Create a copy of the image to draw margin lines
    MarginTest = image.copy()
    
    MarginTest = process_image(MarginTest, brightness_factor, contrast_factor, left_margin_percent, top_margin_percent, vsplit_percent, hsplit_percent)

    # Draw margin lines
    cv2.line(MarginTest, (LeftMargin, TopMargin), (RightMargin, TopMargin), (255, 0, 0), 1)
    cv2.line(MarginTest, (LeftMargin, TopMargin), (LeftMargin, BottomMargin), (255, 0, 0), 1)
    cv2.line(MarginTest, (LeftMargin, BottomMargin), (RightMargin, BottomMargin), (255, 0, 0), 1)
    cv2.line(MarginTest, (RightMargin, TopMargin), (RightMargin, BottomMargin), (255, 0, 0), 1)
    
    # Draw split lines
    # Apply contrast adjustment
    if vsplit_percent != 0:
        cv2.line(MarginTest, (VSplit, 0), (VSplit, height), (0, 0, 255), 1)
    if hsplit_percent != 0:
        cv2.line(MarginTest, (0, HSplit), (width, HSplit), (0, 0, 255), 1)
        

    # Display the image with the margins
    plt.figure(figsize=(10, 20))
    plt.imshow(MarginTest, cmap='gray')
    plt.show()
    
    # Ask the user if they are satisfied with the result
    satisfaction = input("Are you satisfied with the outline you see? (y/n): ")
    if satisfaction.lower() == 'y':
        # Create a subfolder 'modified_images' if it doesn't exist
        modified_images_folder = os.path.join(output_folder, 'modified_images')
        os.makedirs(modified_images_folder, exist_ok=True)
        print("Here is an output based on these parameters")
        if vsplit_percent == 0 and hsplit_percent == 0:
            image = blank_margins(image, left_margin_percent, top_margin_percent)
            image = process_image(image, brightness_factor, contrast_factor, left_margin_percent, top_margin_percent, vsplit_percent, hsplit_percent)
            output_path = os.path.join(modified_images_folder, 'modified_' + filename)
            cv2.imwrite(output_path, image)
            plt.figure(figsize=(10, 20))
            plt.imshow(image, cmap='gray')
            plt.show()
            print("Preprocessed images are saved in a subfolder of your output folder called 'modified_images'.")
                       
            
            batch_run = input("Do you want to batch run this preprocessing routine on the entire input folder? (y/n): ")
            if batch_run.lower() =="y":
                valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
                image_files = [filename for filename in os.listdir(input_folder) if any(filename.lower().endswith(ext) for ext in valid_extensions)]

                # Create a tqdm progress bar
                progress_bar = tqdm(total=len(image_files), desc='Processing Images', unit='image')

                for filename in image_files:
                    file_path = os.path.join(input_folder, filename)
                    
                    image = cv2.imread(file_path)
                    image = blank_margins(image, left_margin_percent, top_margin_percent)
                    image = process_image(image, brightness_factor, contrast_factor, left_margin_percent, top_margin_percent, vsplit_percent, hsplit_percent)
                    output_path = os.path.join(modified_images_folder, 'modified_' + filename)
                    cv2.imwrite(output_path, image)
                    
                    # Update the progress bar
                    progress_bar.update(1)

                # Close the progress bar
                progress_bar.close()
            
            return image
        if vsplit_percent !=0:
            image = process_image(image, brightness_factor, contrast_factor, left_margin_percent, top_margin_percent, vsplit_percent, hsplit_percent)
            left_image, right_image = two_split_vert(image, vsplit_percent, left_margin_percent, top_margin_percent, split_padding=100, show_image=True)
            L_output_path = os.path.join(modified_images_folder, 'modified_1_' + filename)
            cv2.imwrite(L_output_path, left_image)
            R_output_path = os.path.join(modified_images_folder, 'modified_2_' + filename)
            cv2.imwrite(R_output_path, right_image)
            print("Preprocessed images are saved in a subfolder of your output folder called 'modified_images'.")
            
            batch_run = input("Do you want to batch run this preprocessing routine on the entire input folder? (y/n): ")
            if batch_run.lower() =="y":
                valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
                image_files = [filename for filename in os.listdir(input_folder) if any(filename.lower().endswith(ext) for ext in valid_extensions)]

                # Create a tqdm progress bar
                progress_bar = tqdm(total=len(image_files), desc='Processing Images', unit='image')

                for filename in image_files:
                    file_path = os.path.join(input_folder, filename)
                    
                    image = cv2.imread(file_path)
                    
                    image = process_image(image, brightness_factor, contrast_factor, left_margin_percent, top_margin_percent, vsplit_percent, hsplit_percent)
                    left_image, right_image = two_split_vert(image, vsplit_percent, left_margin_percent, top_margin_percent, split_padding=100, show_image=False)
                    L_output_path = os.path.join(modified_images_folder, 'modified_1_' + filename)
                    cv2.imwrite(L_output_path, left_image)
                    R_output_path = os.path.join(modified_images_folder, 'modified_2_' + filename)
                    cv2.imwrite(R_output_path, right_image)
                    
                    # Update the progress bar
                    progress_bar.update(1)

                # Close the progress bar
                progress_bar.close()
            
            return left_image, right_image
        if hsplit_percent !=0:
            image = process_image(image, brightness_factor, contrast_factor, left_margin_percent, top_margin_percent, vsplit_percent, hsplit_percent)
            top_image, bottom_image = two_split_horiz(image, hsplit_percent, left_margin_percent, top_margin_percent, split_padding=100, show_image=True)
            T_output_path = os.path.join(modified_images_folder, 'modified_1_' + filename)
            cv2.imwrite(T_output_path, top_image)
            B_output_path = os.path.join(modified_images_folder, 'modified_2_' + filename)
            cv2.imwrite(B_output_path, bottom_image)
            print("Preprocessed images are saved in a subfolder of your output folder called 'modified_images'.")            
            
            
            batch_run = input("Do you want to batch run this preprocessing routine on the entire input folder? (y/n): ")
            if batch_run.lower() =="y":
                valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
                image_files = [filename for filename in os.listdir(input_folder) if any(filename.lower().endswith(ext) for ext in valid_extensions)]

                # Create a tqdm progress bar
                progress_bar = tqdm(total=len(image_files), desc='Processing Images', unit='image')

                for filename in image_files:
                    file_path = os.path.join(input_folder, filename)
                    
                    image = cv2.imread(file_path)
                    
                    image = process_image(image, brightness_factor, contrast_factor, left_margin_percent, top_margin_percent, vsplit_percent, hsplit_percent)
                    top_image, bottom_image = two_split_horiz(image, hsplit_percent, left_margin_percent, top_margin_percent, split_padding=100, show_image=False)
                    T_output_path = os.path.join(modified_images_folder, 'modified_1_' + filename)
                    cv2.imwrite(T_output_path, top_image)
                    B_output_path = os.path.join(modified_images_folder, 'modified_2_' + filename)
                    cv2.imwrite(B_output_path, bottom_image)
                    
                    # Update the progress bar
                    progress_bar.update(1)

                # Close the progress bar
                progress_bar.close()
            
            
            
            return top_image, bottom_image
    else:
        # If not satisfied, stop the script
            print("Current settings are ", default)
            print("You can modify these settings in the modification cell above.")
            print("Suppose you want to change the side margins to be 8% of the page. Then type: pp.default['left_margin_percent'] = 8")









#Define a program that splits an image vertically into two separate images and adds white space where the split occurs




def two_split_vert(image, vsplit_percent, left_margin_percent, top_margin_percent, split_padding=100, show_image=True):
    # Get the height and width of the input image
    height, width = image.shape[0:2]

    # Calculate margin values based on percentages
    LeftMargin = int(width * (left_margin_percent / 100))
    TopMargin = int(height * (top_margin_percent / 100))
    RightMargin = int(width - LeftMargin)
    BottomMargin = int(height - TopMargin)

    # Insert the desired color for the rectangle (white in this case)
    White = (255, 255, 255)

    # Define rectangles for the portions outside of the margins
    BLMargin = (0, BottomMargin)
    BRCorner = (width, height)
    TLMargin = (LeftMargin, 0)
    TRCorner = (width, TopMargin)
    BRMargin = (RightMargin, height)

    # White out the portions outside of the margins
    image = cv2.rectangle(image, BLMargin, BRCorner, White, -1)
    image = cv2.rectangle(image, TLMargin, BLMargin, White, -1)
    image = cv2.rectangle(image, TLMargin, TRCorner, White, -1)
    image = cv2.rectangle(image, BRMargin, TRCorner, White, -1)

    # Crop the image to give the illusion of normal page margins
    image = image[int(1 - BottomMargin * 1.1):int(BottomMargin * 1.1), int(1 - RightMargin * 1.1):int(RightMargin * 1.1)]

    height, width = image.shape[0:2]
    
    # Split the image into two separate images with padding
    middle = int(width * (vsplit_percent / 100))
    left_image = image[:, :middle]
    right_image = image[:, middle:]

    # Add padding to the split
    padding = np.zeros((height, split_padding, 3), dtype=np.uint8)
    padding = 255 - padding
    left_image = np.hstack((left_image, padding))
    right_image = np.hstack((padding, right_image))

    if show_image:
        # Display the images
        plt.figure(figsize=(10, 20))
        plt.subplot(121), plt.imshow(left_image), plt.title('Left Image')
        plt.subplot(122), plt.imshow(right_image), plt.title('Right Image')
        plt.show()

    return left_image, right_image

# Example usage:
# Load your image and call the function with it
# image = cv2.imread('your_image.png')
# left_image, right_image = white_out_and_crop(image, left_margin_percent=30, top_margin_percent=5, split_padding=100, show_image=True)





#Define a program that splits an image horizontally into two separate images and adds white space where the split occurs

def two_split_horiz(image, hsplit_percent, left_margin_percent, top_margin_percent, split_padding=100, show_image=True):
    # Get the height and width of the input image
    height, width = image.shape[0:2]

    # Calculate margin values based on percentages
    TopMargin = int(height * (top_margin_percent / 100))
    BottomMargin = int(height - TopMargin)
    LeftMargin = int(width * (left_margin_percent / 100))
    RightMargin = int(width - LeftMargin)

    # Insert the desired color for the rectangle (white in this case)
    White = (255, 255, 255)

    # Define rectangles for the portions outside of the margins
    BLMargin = (0, BottomMargin)
    BRCorner = (width, height)
    TLMargin = (LeftMargin, 0)
    TRCorner = (width, TopMargin)
    BRMargin = (RightMargin, height)

    # White out the portions outside of the margins
    image = cv2.rectangle(image, BLMargin, BRCorner, White, -1)
    image = cv2.rectangle(image, TLMargin, BLMargin, White, -1)
    image = cv2.rectangle(image, TLMargin, TRCorner, White, -1)
    image = cv2.rectangle(image, BRMargin, TRCorner, White, -1)

    # Crop the image to give the illusion of normal page margins
    image = image[int(1 - BottomMargin * 1.1):int(BottomMargin * 1.1), int(1 - RightMargin * 1.1):int(RightMargin * 1.1)]

    height, width = image.shape[0:2]
    
    # Split the image into two separate images with padding
    middle = int(height * (hsplit_percent / 100))
    top_image = image[:middle, :]
    bottom_image = image[middle:, :]

    # Add padding to the split
    padding = np.zeros((split_padding, width, 3), dtype=np.uint8)
    padding = 255 - padding
    top_image = np.vstack((top_image, padding))
    bottom_image = np.vstack((padding, bottom_image))

    if show_image:
        # Display the images
        plt.figure(figsize=(10, 20))
        plt.subplot(121), plt.imshow(top_image), plt.title('Top Image')
        plt.subplot(122), plt.imshow(bottom_image), plt.title('Bottom Image')
        plt.show()

    return top_image, bottom_image

# Example usage:
# Load your image and call the function with it
# image = cv2.imread('your_image.png')
# top_image, bottom_image = white_out_and_crop(image, top_margin_percent=5, left_margin_percent=30, split_padding=100, show_image=True)

        
def textract_boxes(image, words_entity_list, table_list, show_image=True):
    # Get image dimensions
    height, width, _ = image.shape
    

    # Iterate over word entities and draw rectangles
    for word_entity in words_entity_list:
        # Accessing the 'BoundingBox' attribute directly
        bounding_box = word_entity.bbox
        
        xmin = int(bounding_box.x * width)
        ymin = int(bounding_box.y * height)
        xmax = int((bounding_box.x + bounding_box.width) * width)
        ymax = int((bounding_box.y + bounding_box.height) * height)

        # Draw a rectangle on the image
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)

    if show_image:
        # Display the image with bounding boxes
        plt.figure(figsize=(10, 20))
        plt.imshow(image)
        plt.show()
        
        satisfaction = input("Do you want to see the table output? (y/n): ")
        if satisfaction.lower() == 'y':
                for table in table_list:
                    df=table.to_pandas()
                    display(df)
        else:
            print('no')








def gcloud_boxes(image, output_file, show_image, save_text_to_txt):
    # Initialize the Google Cloud Vision client
    client = vision.ImageAnnotatorClient()

    # Convert the input image to encoded PNG format
    _, encoded_image = cv2.imencode('.png', image)

    # Create a Vision API image object
    api_image = vision.Image(content=encoded_image.tobytes())

    # Perform text detection
    response = client.text_detection(image=api_image)
    texts = response.text_annotations

    for text in texts:
        vertices = np.array([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])

        # Calculate bounding box coordinates
        xmin, xmax = min(vertices[:, 0]), max(vertices[:, 0])
        ymin, ymax = min(vertices[:, 1]), max(vertices[:, 1])

        # Draw bounding boxes on the image
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)

    if response.error.message:
        print(response.error.message)

    if show_image:
        # Display the image with bounding boxes
        plt.figure(figsize=(10, 20))
        plt.imshow(image)
        plt.show()

    bounding_boxes = []

    # Extract bounding box and text information
    for text in texts:
        vertices = text.bounding_poly.vertices
        box_data = {
            "text": text.description,
            "bounding_box": {
                "vertices": [
                    {"x": vertex.x, "y": vertex.y}
                    for vertex in vertices
                ]
            }
        }
        bounding_boxes.append(box_data)

    # Save the bounding box data to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(bounding_boxes, json_file, indent=2)

    if save_text_to_txt:
        # Save the extracted text to a .txt file
        with open(output_file.replace('.json', '.txt'), 'w') as txt_file:
            for text in texts:
                txt_file.write(text.description + '\n')








# Define a function to adjust brightness
def adjust_brightness(image, brightness_factor):
    # Adjust brightness using cv2.convertScaleAbs
    adjusted_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    return adjusted_image

# Define a function to adjust contrast
def adjust_contrast(image, contrast_factor):
    # Adjust contrast using cv2.convertScaleAbs
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
    return adjusted_image

# Define a function to convert the image to grayscale
def grayscale(image):
    # Convert the image to grayscale
    # Split the image into its color channels
    b, g, r = cv2.split(image)

    # Compute the per-channel means
    b_mean = b.mean()
    g_mean = g.mean()
    r_mean = r.mean()

    # Compute the scaling factors
    k = (b_mean + g_mean + r_mean) / 3

    # Apply the scaling factors to each channel
    corrected_b = (b * k / b_mean).clip(0, 255).astype('uint8')
    corrected_g = (g * k / g_mean).clip(0, 255).astype('uint8')
    corrected_r = (r * k / r_mean).clip(0, 255).astype('uint8')

    # Merge the corrected channels
    corrected_image = cv2.merge((corrected_b, corrected_g, corrected_r))
    
    # Convert the corrected image to grayscale
    gray_image = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    return gray_image

#Whiting out margins and cropping 


def blank_margins(image, left_margin_percent, top_margin_percent):
    # Get the height and width of the input image
    height, width = image.shape[0:2]

    # Calculate margin values based on percentages
    LeftMargin = int(width * (left_margin_percent / 100))
    TopMargin = int(height * (top_margin_percent / 100))
    RightMargin = int(width - LeftMargin)
    BottomMargin = int(height - TopMargin)

    # Insert the desired color for the rectangle (white in this case)
    White = (255, 255, 255)

    # Define rectangles for the portions outside of the margins
    BLMargin = (0, BottomMargin)
    BRCorner = (width, height)
    TLMargin = (LeftMargin, 0)
    TRCorner = (width, TopMargin)
    BRMargin = (RightMargin, height)

    # White out the portions outside of the margins
    image = cv2.rectangle(image, BLMargin, BRCorner, White, -1)
    image = cv2.rectangle(image, TLMargin, BLMargin, White, -1)
    image = cv2.rectangle(image, TLMargin, TRCorner, White, -1)
    image = cv2.rectangle(image, BRMargin, TRCorner, White, -1)

    # Crop the image to give the illusion of normal page margins
    image = image[int(1 - BottomMargin * 1.1):int(BottomMargin * 1.1), int(1 - RightMargin * 1.1):int(RightMargin * 1.1)]

    return image

# Define a function to process an image with options to show the output
def process_image(image, brightness_factor, contrast_factor, left_margin_percent, top_margin_percent, vsplit_percent, hsplit_percent):
    
    # Convert to grayscale if needed
    image = grayscale(image)   

    # Apply brightness adjustment
    if brightness_factor != 1.0:
        image = adjust_brightness(image, brightness_factor)

    # Apply contrast adjustment
    if contrast_factor != 1.0:
        image = adjust_contrast(image, contrast_factor)
    
    return image



