#! usr/bin/env python3
# Andrew Johnson
# January 2017

"""
Generates clusters of random points with a Gaussian distribution belonging
to one of two labels (red or blue), such that they cannot be classified
by a linear classifier. Using keras, creates a feedforward neural network
that classifies them with high accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import pylab


def generate_data(size = 100, train_percent = 75):
    """Randomly generates 4 clusters of 2 colors and displays a plot
    Args:
        size: Number of points in each cluster
        train_percent: what percent to use for training
            save for vs. testing
    Returns: A tuple containing the following:
        training set data, training set labels, 
        testing set data, testing set labels
    """
    # Randomly choose points for 4 clusters of 2 colors
    stdv = 0.6
    x_red  = np.random.normal(loc=0, scale=stdv, size=size)
    y_red  = np.random.normal(loc=0, scale=stdv, size=size)
    x_blue = np.random.normal(loc=-4, scale=stdv, size=size)
    y_blue = np.random.normal(loc=0, scale=stdv, size=size)
    x_blue = np.append(x_blue, np.random.normal(loc=4, scale=stdv, size=size))
    y_blue = np.append(y_blue, np.random.normal(loc=0, scale=stdv, size=size))
    x_blue = np.append(x_blue, np.random.normal(loc=0, scale=stdv, size=size))
    y_blue = np.append(y_blue, np.random.normal(loc=-4, scale=stdv, size=size))

    # Plot the points (training & testing combined here)
    plt.scatter(x_red, y_red, color="red")
    plt.scatter(x_blue, y_blue, color="blue")
    # Show axes
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.show()

    # Combine x and y coordinates for each color
    red = list(zip(x_red, y_red))
    blue = list(zip(x_blue, y_blue))

    # Add color tag and shuffle all together
    labeled = []
    for entry in red:
        labeled += [(entry, 1)]
    for entry in blue:
        labeled += [(entry, 0)]
    np.random.shuffle(labeled)

    # Separate shuffled data from labels
    data, labels = zip(*labeled)
    data = np.array(data)
    labels = np.array(labels)

    # Reserve some data for testing
    train_num = int(len(data)* train_percent / 100)
    training_data = data[:train_num]
    testing_data = data[train_num:]
    training_labels = labels[:train_num]
    testing_labels = labels[train_num:]

    return (training_data, training_labels, testing_data, testing_labels)



def neural_model(all_data, epochs=200, batch=16, display=True, first_HL_nodes=4, second_HL=False, act_func="relu"):
    """ Trains a neural network on given data and labels, returns accuracy 
        and optionally generates a display of that the model looks like.
    """
    training_data, training_labels, testing_data, testing_labels = all_data
    
    # Set up model
    my_model = Sequential()
    my_model.add(Dense(first_HL_nodes, input_dim=2, activation=act_func)) # 2 inputs, 1st HL
    if second_HL:
        my_model.add(Dense(3, activation=act_func)) # Optional 2nd HL with 3 nodes
    my_model.add(Dense(1, activation=act_func)) #1 output 

    # Compile model
    my_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit model
    my_model.fit(training_data, training_labels, nb_epoch=epochs, batch_size=batch, verbose=0)

    # Optionally visualize model
    if display:
        # Generate random points
        rand_size = 3000 # The "resolution" of the graphical representation of the model
        rand_x = np.random.uniform(low=-6, high=6, size=(rand_size,))
        rand_y = np.random.uniform(low=-7, high=2, size=(rand_size,))
        rand_xy = [[rand_x[i],rand_y[i]] for i in range(len(rand_x))]
        rand_xy = np.array(rand_xy)

        # Use model to predict a label for each point
        predict_labels = my_model.predict(rand_xy) 

        red_x_guesses = []
        red_y_guesses = []
        blue_x_guesses = []
        blue_y_guesses = []

        # Apply step function
        for i in range(len(predict_labels)):
            if predict_labels[i] < 0.5:
                blue_x_guesses += [rand_xy[i][0]]
                blue_y_guesses += [rand_xy[i][1]]
            else:
                red_x_guesses += [rand_xy[i][0]]
                red_y_guesses += [rand_xy[i][1]]

        # Show plot
        plt.scatter(red_x_guesses, red_y_guesses, color="red")
        plt.scatter(blue_x_guesses, blue_y_guesses, color="blue")
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')

        pylab.xlim([-6,6])
        pylab.ylim([-7,2])
        plt.show()
        
    # Evaluate model
    percentage = my_model.evaluate(testing_data, testing_labels, batch_size=len(testing_data), verbose=0) # Returns [loss, accuracy]
    percentage = percentage[1] * 100  # Accuracy %
    return percentage



if __name__ == "__main__":
    all_data = generate_data()
    neural_model(all_data, act_func="sigmoid", second_HL=True)
    
