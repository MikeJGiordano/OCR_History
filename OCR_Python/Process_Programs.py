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

def Textract_process_image(image_path, json_output):
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

    # Show the modified image with blue bounding boxes
    plt.figure(figsize=(10, 20))
    plt.imshow(image)
    plt.show()


#Set default dictionarys
vsplit = {
    "left_margin_percent" : 30, 
    "top_margin_percent" : 5,
    "vsplit_percent" : 50,
    "hsplit_percent" : 0,
    "brightness_factor" : 1,
    "contrast_factor" : 1
}

hsplit = {
    "left_margin_percent" : 30, 
    "top_margin_percent" : 5,
    "vsplit_percent" : 0,
    "hsplit_percent" : 50,
    "brightness_factor" : 1,
    "contrast_factor" : 1
}

full = {
    "left_margin_percent":15,
    "top_margin_percent":15,
    "vsplit_percent":0,
    "hsplit_percent":0,
    "brightness_factor" : 1,
    "contrast_factor" : 1
}


#Hone the correct split points and margin lines

def draw_margins_splits(image, left_margin_percent, top_margin_percent, vsplit_percent, hsplit_percent, brightness_factor, contrast_factor):
    # Get the height and width of the input image
    height, width = image.shape[0:2]

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
        print("Here is an output based on these parameters")
        if vsplit_percent == 0 and hsplit_percent == 0:
            image = blank_margins(image, left_margin_percent, top_margin_percent)
            image = process_image(image, brightness_factor, contrast_factor, left_margin_percent, top_margin_percent, vsplit_percent, hsplit_percent)
            plt.figure(figsize=(10, 20))
            plt.imshow(image, cmap='gray')
            plt.show()
            return image
        if vsplit_percent !=0:
            image = process_image(image, brightness_factor, contrast_factor, left_margin_percent, top_margin_percent, vsplit_percent, hsplit_percent)
            left_image, right_image = two_split_vert(image, vsplit_percent, left_margin_percent, top_margin_percent, split_padding=100, show_image=True)
            return left_image, right_image
        if hsplit_percent !=0:
            image = process_image(image, brightness_factor, contrast_factor, left_margin_percent, top_margin_percent, vsplit_percent, hsplit_percent)
            top_image, bottom_image = two_split_horiz(image, hsplit_percent, left_margin_percent, top_margin_percent, split_padding=100, show_image=True)
            return top_image, bottom_image
    else:
        # If not satisfied, stop the script
        if vsplit_percent == 0 and hsplit_percent == 0:
            print("Current settings are ", full)
        if vsplit_percent !=0:
            print("Current settings are ", vsplit)
        if hsplit_percent !=0:
            print("Current settings are ", hsplit)


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


#Define a function that shows the google vision image output for one image
def gcloud_boxes(image, output_file, show_image=True):
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

# Example usage:
# Load your image using cv2 or another method
# image = cv2.imread('input_image.jpg')

# Specify the output JSON file
# detect_and_save_text(image, 'output.json', show_image=False)


#Define a function that shows the google vision image output for two images

def gcloud_boxes_two_images(image1, image2, output_file1, output_file2, show_image=True):
    def detect_and_save_text(image, output_file):
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

    # Detect and save text for the first image
    detect_and_save_text(image1, output_file1)

    # Detect and save text for the second image
    detect_and_save_text(image2, output_file2)
    
    if show_image:
        # Display the images
        plt.figure(figsize=(10, 20))
        plt.subplot(121), plt.imshow(image1), plt.title('Left Image')
        plt.subplot(122), plt.imshow(image2), plt.title('Right Image')
        plt.show()

    return image1, image2

# Example usage:
# Load your images using cv2 or another method
# image1 = cv2.imread('input_image1.jpg')
# image2 = cv2.imread('input_image2.jpg')

# Specify the output JSON files for both images
# gcloud_boxes_for_two_images(image1, image2, 'output1.json', 'output2.json', show_image=False)

#Define where you want to output the json file





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



def batch_process(input_folder, output_folder, left_margin_percent, top_margin_percent, vsplit_percent, hsplit_percent, brightness_factor, contrast_factor):
    """
    Batch process images in the input_folder and save the processed images to the output_folder.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where processed images will be saved.
        parameters (dict): Parameters to pass to the image processing function.
    """
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, 'processed_' + filename)  # Add 'processed_' to the output filename
            L_output_path = os.path.join(output_folder, 'L_processed_' + filename)  # Add 'processed_' to the output filename
            R_output_path = os.path.join(output_folder, 'R_processed_' + filename)  # Add 'processed_' to the output filename
            T_output_path = os.path.join(output_folder, 'T_processed_' + filename)  # Add 'processed_' to the output filename
            B_output_path = os.path.join(output_folder, 'B_processed_' + filename)  # Add 'processed_' to the output filename
            image = cv2.imread(input_path)
            image = process_image(image, brightness_factor, contrast_factor, left_margin_percent, top_margin_percent, vsplit_percent, hsplit_percent)
            if vsplit_percent == 0 and hsplit_percent == 0:
               image = blank_margins(image, left_margin_percent, top_margin_percent)
               cv2.imwrite(output_path, image)
            if vsplit_percent !=0:
                left_image, right_image = two_split_vert(image, vsplit_percent, left_margin_percent, top_margin_percent, split_padding=100, show_image=False)
                cv2.imwrite(L_output_path, left_image)
                cv2.imwrite(R_output_path, right_image)
            if hsplit_percent !=0:
                top_image, bottom_image = two_split_horiz(image, hsplit_percent, left_margin_percent, top_margin_percent, split_padding=100, show_image=False)
                cv2.imwrite(T_output_path, top_image)
                cv2.imwrite(B_output_path, bottom_image)

    






