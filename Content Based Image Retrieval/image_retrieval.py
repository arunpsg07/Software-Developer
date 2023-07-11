import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import hog, daisy
from skimage import data, color, exposure
from scipy.spatial.distance import euclidean


# Function to extract features from an image
def extract_features(image_path):

    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image
    resized = cv2.resize(gray, (480, 480))

    # Gabor filter
    ksize = 9
    sigma = 4.0
    theta = 1.0
    lambd = 4.0
    gamma = 0.5
    gabor = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma)
    filtered = cv2.filter2D(resized, cv2.CV_8UC3, gabor)

    # HOG feature
    fd, _ = hog(resized, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)

    # RGB histogram feature
    bins = 16
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    hist_flat = hist.flatten()

    # Daisy feature
    desc = daisy(resized, step=180, radius=58, rings=2, histograms=6, orientations=8, visualize=False)
    descs = desc.reshape(desc.shape[0]*desc.shape[1], desc.shape[2])

    # Concatenate the feature vectors
    features = np.concatenate((filtered.flatten(), fd, hist_flat, descs.flatten()))

    return features

# Function to compare two images
def compare_images(image1_path, image2_path):

    # Extract features from the images
    features1 = extract_features(image1_path)
    features2 = extract_features(image2_path)


    # Compute the Euclidean distance between the feature vectors
    dist = euclidean(features1, features2)

    return dist


def operation():

    # Path to the directory containing the images
    directory = r"E:\academic lab\CV\package\databases/"

    # Path to the query image
    query_path = r"E:\academic lab\CV\package\upload_folder\query_image.png"

    # List to store the distances between the query image and the database images
    distances = []

    # Loop over the database images and compare them with the query image
    for filename in os.listdir(directory):

        if filename.endswith('.png') :
        
            image_path = os.path.join(directory, filename)
            dist = compare_images(query_path, image_path)
            distances.append((filename, dist))

    # Sort the distances in ascending order
    distances.sort(key=lambda x: x[1])

    image_path = []

    for i in range(3):
        image_path.append( os.path.join( directory , distances[i][0] ) )
    
    return image_path
