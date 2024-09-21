import cv2
import numpy as np
from screeninfo import get_monitors
import pytesseract
from pytesseract import Output
import os
from open_ai import Annotate
import difflib  # For matching phrases with a high similarity score


# Define keywords and their annotations here
# keywords = {'A green and yellow parrot': 'Parrot speaks different languages', 'Mr Pontellier unable to': 'Annoyed by noisy birds', 'The day was Sunday': 'Mr Pontellier reads newspaper', 'He was already acquainted': 'Familiar with market reports', 'Two young girls': 'Farival twins playing piano'}


# Image processing functions
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_contrast(image, clipLimit=2.0, tileGridSize=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(image)


def apply_morphology(image):
    kernel = np.ones((1, 1), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def reduce_noise(image):
    return cv2.fastNlMeansDenoising(image, None, 30, 7, 21)


def thresholding(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)


def get_screen_resolution():
    monitor = get_monitors()[0]
    return monitor.width, monitor.height


def draw_label_old(image, text, position, bg_color, text_color, font_scale, thickness):
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    # Adjust the box_coords to make sure text fits inside the rectangle
    box_coords = ((position[0], position[1] - text_height - 10), (position[0] + text_width + 10, position[1]))
    cv2.rectangle(image, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(image, text, (position[0] + 5, position[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                thickness)


def draw_label(annotated_image, label_text, x, y, annotation_height, font_scale, thickness):
    (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    # Ensure label is within bounds of the left margin
    y_start = max(0, y - label_height)
    y_end = min(annotated_image.shape[0], y_start + label_height)

    # Draw label background
    cv2.rectangle(annotated_image, (x, y_start), (x + label_width, y_end), (255, 255, 255), cv2.FILLED)
    # Draw label text
    cv2.putText(annotated_image, label_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)


# Function to adjust annotation position if overlapping
def adjust_annotation_position(y, occupied_positions, annotation_height):
    if not occupied_positions:  # If no positions are occupied, return original y
        return y
    for pos in occupied_positions:
        if abs(y - pos) < annotation_height:  # Check for overlap
            return adjust_annotation_position(y + annotation_height, occupied_positions,
                                              annotation_height)  # Adjust position
    return y


def find_best_phrase_match(ocr_words, ocr_indices, phrase):
    phrase_words = phrase.split()
    max_matches = 0
    best_match_start_index = -1
    best_match_end_index = -1

    for i in range(len(ocr_words) - len(phrase_words) + 1):
        matches = sum(1 for j in range(len(phrase_words)) if phrase_words[j].lower() in ocr_words[i + j].lower())
        if matches > max_matches:
            max_matches = matches
            best_match_start_index = ocr_indices[i]  # Store the original start index
            best_match_end_index = ocr_indices[i + len(phrase_words) - 1]  # Store the original end index

    return best_match_start_index, best_match_end_index, max_matches


def get_full_text_and_bounding_box(img, original, options):
    view, extra, annotations_word_length, annotations_per_page = options
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    extra_space = 400  # Additional space for annotations
    # Adjust the width of the annotated_image to add extra_space on both sides
    annotated_image = np.zeros((img.shape[0], img.shape[1] + 2 * extra_space, 3), dtype=np.uint8)
    # Copy the original image into the center of the annotated_image, accounting for the extra space on both sides
    annotated_image[:, extra_space:-extra_space] = original.copy()

    # Lists to hold OCR words and their original indices
    ocr_words = []
    ocr_indices = []
    for i, word in enumerate(d['text']):
        if int(d['conf'][i]) > 60:
            ocr_words.append(word.lower())
            ocr_indices.append(i)

    full_text = " ".join(ocr_words)
    a = Annotate()
    keywords = a.get_annotations(text_to_annotate=full_text, word_length=annotations_word_length,
                                 annotations_per_page=annotations_per_page, extra=extra)
    if not keywords:
        print(False)
    # print("Full text extracted from the image:\n", full_text)

    annotation_height = 30  # Adjust based on your font size
    font_scale = 0.5
    thickness = 1
    annotation_counter = 1

    occupied_positions = []  # Keep track of occupied vertical positions for annotations
    for keyword, annotation in keywords.items():
        start_index, end_index, matches = find_best_phrase_match(ocr_words, ocr_indices, keyword)

        # View mode "1" - avoid overlaps
        if view == "1" and matches > 0 and start_index != -1 and end_index != -1:
            x_start, y_start, w_start, h_start = d['left'][start_index], d['top'][start_index], d['width'][start_index], \
                d['height'][start_index]
            x_end, y_end, w_end, h_end = d['left'][end_index], d['top'][end_index], d['width'][end_index], d['height'][
                end_index]

            # Calculate encompassing bounding box coordinates
            x = x_start + extra_space
            y = min(y_start, y_end)
            w = (x_end + w_end) - x_start
            h = max(y_start + h_start, y_end + h_end) - y

            adjusted_y = adjust_annotation_position(y, occupied_positions, annotation_height)
            occupied_positions.append(adjusted_y)

            cv2.rectangle(annotated_image, (x, adjusted_y), (x + w, adjusted_y + h), (0, 0, 0), 2)
            draw_label(annotated_image, f"\"{keyword}\"-{annotation}", x - 100, adjusted_y + h // 2, annotation_height,
                       font_scale,
                       thickness)

        # View mode "2" - direct placement next to bounding box
        elif matches > 0 and start_index != -1 and end_index != -1:
            x_start, y_start, w_start, h_start = d['left'][start_index], d['top'][start_index], d['width'][start_index], \
                d['height'][start_index]
            x_end, y_end, w_end, h_end = d['left'][end_index], d['top'][end_index], d['width'][end_index], d['height'][
                end_index]

            # Calculate encompassing bounding box coordinates
            x = x_start + extra_space
            y = min(y_start, y_end)
            w = (x_end + w_end) - x_start
            h = max(y_start + h_start, y_end + h_end) - y

            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 0, 0), 2)
            draw_label_old(annotated_image, f"{annotation_counter}. {annotation}", (5, y), (255, 255, 255), (0, 0, 0),
                           font_scale, thickness)
            annotation_counter += 1

    return annotated_image


# Assuming the image processing functions and everything else are defined above...

def process_images_from_directory(input_dir, output_dir, options):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all image files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Process the image
            try:
                image = cv2.imread(input_path)
                if image is None:
                    print(f"Error: Could not read image {input_path}. Skipping...")
                    continue

                screen_width, screen_height = get_screen_resolution()
                new_height = screen_height - 100
                aspect_ratio = image.shape[1] / image.shape[0]
                new_width = int(new_height * aspect_ratio)
                resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

                grayscale = get_grayscale(resized_image)

                contrast_enhanced = enhance_contrast(grayscale)
                noise_reduced = reduce_noise(contrast_enhanced)
                thresholded = thresholding(noise_reduced)
                morphology_applied = apply_morphology(thresholded)

                processed_image = get_full_text_and_bounding_box(morphology_applied, resized_image, options)

                # Save the processed image
                cv2.imwrite(output_path, processed_image)
                print(f"Processed image saved to {output_path}")

            except Exception as e:
                print(f"Failed to process {input_path}: {e}")


if __name__ == '__main__':
    view = "1"
    # extra = "This is a Feminist piece of writing, link the annotations back to feminism. "
    extra = "Please Act as me and read this and add annotations on how you(I) would feel while reading it and how he " \
            "engages the reader and how he inflicts emotions"
    annotations_word_length = 8
    annotations_per_page = 8

    options = view, extra, annotations_word_length, annotations_per_page
    # Example usage - specify the paths to your directories here
    input_dir = 'input_photos'  # Update this path
    output_dir = 'output_photos'  # Update this path
    process_images_from_directory(input_dir, output_dir, options=options)
    input("E?")
