import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours in the image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (presumably the Sudoku grid)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the corners of the largest contour
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
    
    # Warp the image to a top-down perspective (optional, depending on the angle of the input image)
    sudoku_warped = warp_image(image, approx.reshape(4, 2))
    
    # Split the Sudoku grid into individual cells
    grid = split_into_cells(sudoku_warped)
    
    # Recognize digits in each cell using the digit recognition model
    sudoku_grid = recognize_digits(grid)
    
    return sudoku_grid

def warp_image(image, corners):
    # Perform perspective transform to get a top-down view of the Sudoku grid
    # Calculate the destination points (a fixed size grid)
    dst_points = np.array([[0, 0], [450, 0], [450, 450], [0, 450]], dtype=np.float32)
    
    # Calculate the transformation matrix
    M = cv2.getPerspectiveTransform(corners, dst_points)
    
    # Apply the perspective transform to get a top-down view of the Sudoku grid
    warped = cv2.warpPerspective(image, M, (450, 450))
    
    return warped

def split_into_cells(image):
    # Split the Sudoku grid into individual cells
    grid = []
    cell_size = image.shape[0] // 9
    for i in range(9):
        for j in range(9):
            cell = image[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            grid.append(cell)
    return grid

def recognize_digits(grid):
    # Recognize digits in each cell using the digit recognition model
    sudoku_grid = []
    for cell in grid:
        # Preprocess the cell image
        cell = preprocess_cell(cell)
        
        # Predict the digit using the digit recognition model
        digit = model.predict_classes(cell.reshape(1, 28, 28, 1))
        sudoku_grid.append(digit[0])
    return sudoku_grid

def preprocess_cell(cell):
    # Resize the cell to match the input size of the digit recognition model
    cell = cv2.resize(cell, (28, 28))
    
    # Convert the cell to grayscale
    cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    
    # Threshold the cell image to binarize it
    _, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Normalize the pixel values
    cell = cell / 255.0
    
    # Reshape the cell to match the input shape of the digit recognition model
    cell = np.reshape(cell, (28, 28, 1))
    
    return cell

# Load digit recognition model (trained neural network, CNN for example)
model = load_model('digit_recognition_model.h5')

# Example usage:
image_path = 'sudoku_image.jpg'
sudoku_grid = preprocess_image(image_path)
print(sudoku_grid)
