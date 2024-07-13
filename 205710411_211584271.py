# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:35:53 2024

@author: kfira
"""

import matplotlib.pyplot as plt
import random
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

# The digits dataset
# The data consisits of 8x8 images of digits
digits = datasets.load_digits()
nDigits = digits.data.shape[0]
#Plot 9 random images from dataset
rows = 3
cols = 3
fig, axs = plt.subplots(rows, cols)
for i in range(rows):
    for j in range(cols):
        index = random.randint(0, nDigits -1)
        axs[i, j].imshow(digits.images[index], cmap='gray')
        axs[i, j].axis('off')
        axs[i, j].set_title(f'real: {digits.target[index]}', fontdict = {'fontsize' : 10})
plt.show()

#split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.5)

#Create and train a support vector classifier
classifier = svm.SVC(kernel='rbf', gamma = 'scale')
classifier.fit(X_train, y_train)

#predict the values on the test set
predicted = classifier.predict(X_test)

# Plot 9 random predictions
nTest = X_test.shape[0]
fig, axs = plt.subplots(rows, cols)
for i in range(rows):
    for j in range(cols):
        index = random.randint(0 , nTest - 1)
        image = X_test[index].reshape(8, 8)
        axs[i, j].imshow(image, cmap='gray')
        axs[i, j].axis('off')
        axs[i, j].set_title(f'Pred: {predicted[index]}, Real: {y_test[index]}', fontdict={'fontsize': 10})
plt.show()

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, predicted)

# Find the digit with the highest true predictions and print its % of errores 
true_predictions = np.diag(conf_matrix)
most_true_digit = np.argmax(true_predictions)
total_most_true_digit = np.sum(conf_matrix[most_true_digit])
errors_most_true_digit = total_most_true_digit - true_predictions[most_true_digit]
error_percentage_most_true_digit = (errors_most_true_digit / total_most_true_digit) * 100
print(f"Digit with the highest true predictions: {most_true_digit}")
print(f"Percentage of errors: {error_percentage_most_true_digit:.2f}%")

# Find the digit with the lowest true predictions and print its % of errores
least_true_digit = np.argmin(true_predictions)
total_least_true_digit = np.sum(conf_matrix[least_true_digit])
errors_least_true_digit = total_least_true_digit - true_predictions[least_true_digit]
error_percentage_least_true_digit = (errors_least_true_digit / total_least_true_digit) * 100
print(f"Digit with the lowest true predictions: {least_true_digit}")
print(f"Percentage of errors: {error_percentage_least_true_digit:.2f}%")

# Plot at least 2 different digits with a true and false prediction
correct_indices = np.where(predicted == y_test)[0]
incorrect_indices = np.where(predicted != y_test)[0]
selected_correct_indices = np.random.choice(correct_indices, 2, replace = False)
selected_incorrect_indices = np.random.choice(incorrect_indices, 2, replace=False)
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
indices = np.concatenate((selected_correct_indices, selected_incorrect_indices))
for i, ax in enumerate(axs.flat):
    index = indices[i]
    image = X_test[index].reshape(8, 8)
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    true_label = y_test[index]
    predicted_label = predicted[index]
    ax.set_title(f'True: {true_label}, Pred: {predicted_label}', fontdict={'fontsize': 10})
plt.show()

# Function to get correctly predicted digit samples
def get_correct_digit_samples(digit):
    """
    Returns samples of a given digit that are correctly predicted by the SVM classifier.

    Parameters:
    digit (int): The digit to be retrieved (between 0 and 9).

    Returns:
    list: A list of correctly predicted digit images.
    """
    correct_indices = np.where((y_test == digit) & (predicted == digit))[0]
    if len(correct_indices) == 0:
        print(f"No correctly predicted samples for digit {digit}. Try another digit.")
        return []
    
    selected_indices = np.random.choice(correct_indices, 10, replace=False)
    images = [X_test[index].reshape(8, 8) for index in selected_indices]
    return images


# Loop to check for another digit if the current one has no correctly predicted samples
def plot_correct_samples_until_success(digit):
    """
    Plots correctly predicted samples for the given digit. If no correct predictions are found,
    it tries the next digit in sequence.

    Parameters:
    digit (int): The starting digit to be plotted (between 0 and 9).
    """
    original_digit = digit
    while True:
        print(f"Trying digit {digit}...")
        correct_indices = np.where((y_test == digit) & (predicted == digit))[0]
        if len(correct_indices) > 0:
            get_correct_digit_samples(digit)
            break
        else:
            print(f"No correctly predicted samples for digit {digit}. Trying next digit.")
            digit = (digit + 1) % 10
            if digit == original_digit:
                print("No correctly predicted samples for any digit. Exiting.")
                break

plot_correct_samples_until_success(4)


# Function to plot a date with digit images
def plot_date(date_digits, colors, title):
    """
    Plots the given date digits with specified colors.

    Parameters:
    date_digits (list): List of digit integers representing the date.
    colors (list): List of colors for each part of the date (day, month, year).
    title (str): Title for the plot.
    """
    fig, ax = plt.subplots(figsize=(len(date_digits), 2))
    
    images = []
    for digit in date_digits:
        images.append(get_correct_digit_samples(digit)[0])
    
    concatenated_image = np.concatenate(images, axis=1)
    
    rgb_image = np.zeros((concatenated_image.shape[0], concatenated_image.shape[1], 3))
    
    # Fill the RGB image with the corresponding colors
    offset = 0
    for i, color in enumerate(colors):
        for x in range(8):
            for y in range(8):
                if concatenated_image[x, offset + y] > 0:
                    if color == 'red':
                        rgb_image[x, offset + y, 0] = 1
                    elif color == 'blue':
                        rgb_image[x, offset + y, 2] = 1
                    elif color == 'green':
                        rgb_image[x, offset + y, 1] = 1
        offset += 8
    
    ax.imshow(rgb_image)
    ax.axis('off')
    plt.title(title, fontsize=20)
    plt.show()

# Define colors for different parts of the date
day_color = 'red'
month_color = 'blue'
year_color = 'green'

# Plot Kfir's birthday
kfir_birthday = [1, 3, 0, 8, 1, 9, 9, 4]
kfir_colors = [day_color]*2 + [month_color]*2 + [year_color]*4
plot_date(kfir_birthday, kfir_colors, "Kfir's birthday")

# Plot Adam Kial's birthday
adam_birthday = [2, 3, 0, 2, 2, 0, 0, 1]
adam_colors = [day_color]*2 + [month_color]*2 + [year_color]*4
plot_date(adam_birthday, adam_colors, "Adam Kial's birthday")


