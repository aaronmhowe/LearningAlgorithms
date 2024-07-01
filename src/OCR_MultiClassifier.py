import numpy as np
import os

# data
training_data = []
training_labels = []
testing_data = []
testing_labels = []

in_ocr_test = os.path.join(os.path.dirname(__file__), 'OCR-data/ocr_test.txt')
in_ocr_train = os.path.join(os.path.dirname(__file__), 'OCR-data/ocr_train.txt')

# number of classes
classes = 26

# max number of iterations
iterations = 20

# learning rate
a = 1

"""
Function debugging file reading and writing
@param path: file path of the input file(s)
"""
def debug_file_read(path):
    if os.path.exists(path):
        if os.access(path, os.R_OK):
            # print(f"Input file {path} exists and can be read from.")
            try:
                with open(path, 'r') as f:
                    # print(f"Beginning of input {path}:")
                    for _ in range(5):
                        input_line = f.readline().strip()
                        # print(input_line if input_line else "[Line is empty]")
            except Exception as e:
                print(f"Encountered an issue reading from file: {e}")
        else:
            print(f"Input file {path} exists but cannot be read from.")
    else:
        print(f"Input file {path} does not exist.")


"""
Function for reading the input files and loading the data into memory
@param training_file: input file containing training examples
@param testing_file: input file containing testing examples
"""
def read_files(training_file, testing_file):

    global training_data, training_labels, testing_data, testing_labels

    def process(path):

        data = []
        labels = []

        with open(path, 'r') as file:

            for line in file:
                words = line.strip().split()

                if len(words) > 2 and words[1].startswith('im'):
                    features = [int(pixel) for pixel in words[1][2:]]
                    if len(features) != 128:
                        print(f"Expected 128 features, got {len(features)} in line: {line.strip()}")
                        continue
                    label = ord(words[-2].lower()) - ord('a')
                    data.append(features)
                    labels.append(label)
        return np.array(data), np.array(labels)
    
    training_data, training_labels = process(training_file)
    testing_data, testing_labels = process(testing_file)

    # for debugging
    print(f"Reading data from training data, total of {len(training_data)}")
    training_data, training_labels = process(training_file)
    print(f"Reading data from testing data, total of {len(testing_data)}")
    testing_data, testing_labels = process(testing_file)

    if len(training_data) == 0 or len(testing_data) == 0:
        print("No data found")
        print(f"Training Data File Path: {training_file}")
        print("Training Data:")
        # with open(training_file, 'r') as f:
        #     for _ in range(5):
        #         print(f.readline().strip())

        print(f"Testing Data File Path: {testing_file}")
        print("Testing Data:")
        # with open(testing_file, 'r') as f:
        #     for _ in range(5):
        #         print(f.readline().strip())
                
        print("Working Directory:", os.getcwd())

        if not os.path.exists(training_file):
            print(f"Training file {training_file} does not exist")
        if not os.path.exists(testing_file):
            print(f"Testing file {testing_file} does not exist")


"""
Pre-processing step, normalizing the data and adding a bias
"""
def pre_process_data():

    global training_data, testing_data

    # reshaping training and testing data to two dimensions
    if training_data.ndim == 1:
        training_data = training_data.reshape(-1, 1)
    if testing_data.ndim == 1:
        testing_data = testing_data.reshape(-1, 1)

    # normalizing the data between zero and one
    training_data = training_data.astype(float) / 255.0
    testing_data = testing_data.astype(float) / 255.0

    # adding a bias
    training_data = np.hstack((training_data, np.ones((training_data.shape[0], 1))))
    testing_data = np.hstack((testing_data, np.ones((testing_data.shape[0], 1))))


"""
Function that implements the standard perceptron algorithm
@param data: input data
@param labels: true labels
@param itr: max number of iterations
"""
def multiclass_perceptron(data, labels, itr):

    class_count = classes
    features = data.shape[1]
    w = np.zeros((class_count, features))

    mistake_count = []
    training_accuracy_count = []
    testing_accuracy_count = []

    for iteration in range(itr):
        mistakes = 0

        # computing scores, making predictions, and performing weight updates
        for x, y in zip(data, labels):
            training_scores = np.dot(w, x)
            y_prediction = np.argmax(training_scores)

            # update rule
            if y_prediction != y:
                mistakes += 1
                w[y] += a * x
                w[y] -= a * x

        mistake_count.append(mistakes)

        training_accuracy = compute_accuracy(w, data, labels)
        testing_accuracy = compute_accuracy(w, testing_data, testing_labels)
        training_accuracy_count.append(training_accuracy)
        testing_accuracy_count.append(testing_accuracy)

        dump_output(f"iteration-{iteration+1} {mistakes}\n")
        dump_output(f"iteration-{iteration+1} {training_accuracy:.4f} {testing_accuracy:.4f}\n")

        if mistakes == 0:
            break

    return w



"""
Function for computing the accuracy of the training algorithm per iteration
@param w: weight vector
@param data: training data
@param labels: true labels
"""
def compute_accuracy(w, data, labels):

    accuracy = 0
    total = len(data)

    # print(f"Shape of Weights: {w.shape}")
    # print(f"Shape of Data: {data.shape}")
    # print(f"Shape of Labels: {labels.shape}")

    for x, y in zip(data, labels):
        # print(f"Shape of x: {x.shape}")
        try:
            training_scores = np.dot(w, x)
            y_prediction = np.argmax(training_scores)

            if y_prediction == y:
                accuracy += 1
        except ValueError as e:
            print(f"Error in Computing Accuracy: {e}")
            print(f"Shape of problematic x: {x.shape}")
            print(f"First few elements of x: {x[:5]}")
            raise

    return accuracy / total


"""
Function that implements averaged perceptron
@param data: input data
@param labels: true labels
@param itr: max number of iterations
"""
def averaged_multiclass_perceptron(data, labels, itr):

    features = data.shape[1]
    class_count = classes
    w = np.zeros((class_count, features))
    # new weight vector holding averaged weights
    w_avg = np.zeros((class_count, features))

    mistake_count = []
    training_accuracy_count = []
    testing_accuracy_count = []

    # averaged weight counter
    counting_averages = 1

    for iteration in range(itr):
        mistakes = 0

        for x, y in zip(data, labels):
            training_scores = np.dot(w, x)
            y_prediction = np.argmax(training_scores)

            if y_prediction != y:
                mistakes += 1
                w[y] += a * x
                w[y] -= a * x
            
            # computing averaged weights
            w_avg += counting_averages * w
            counting_averages += 1

        mistake_count.append(mistakes)

        w_avg_count = w_avg / counting_averages

        training_accuracy = compute_accuracy(w_avg_count, data, labels)
        testing_accuracy = compute_accuracy(w_avg_count, testing_data, testing_labels)
        training_accuracy_count.append(training_accuracy)
        testing_accuracy_count.append(testing_accuracy)

        dump_output(f"iteration-{iteration+1} {mistakes}\n")
        dump_output(f"iteration-{iteration+1} {training_accuracy:.4f} {testing_accuracy:.4f}\n")

        if mistakes == 0:
            break

    # final averaged weight calculation
    compute_average_weight = w_avg / counting_averages

    return compute_average_weight


"""
Function to make predictions on training examples
@param w: the weight vector
@param examples: training data
"""
def prediction(w, examples):

    training_scores = np.dot(w, examples)
    training_prediction = np.argmax(training_scores)

    return training_prediction


def predicting_multiple(w, examples):

    training_scores = np.dot(examples, w.T)
    training_prediction = np.argmax(training_scores, axis=1)

    return training_prediction


"""
Function for dumping the results into the output file
@param output: the resulting output data
"""
def dump_output(output):
    
    with open('output.txt', 'a') as f:
        f.write(output)


"""
Main application function
"""
def main():

    global training_data, training_labels, testing_data, testing_labels

    debug_file_read(in_ocr_train)
    debug_file_read(in_ocr_test)

    read_files(in_ocr_train, in_ocr_test)

    # print(f"Shape of Training Data: {training_data.shape}")
    # print(f"Shape of Training Labels: {training_labels.shape}")

    if len(training_data) == 0 or len(testing_data) == 0:
        print("Encountered a problem loading data into memory, no data loaded.")
        return

    pre_process_data()

    print("Training the model using standard perceptron...")
    weights = multiclass_perceptron(training_data, training_labels, iterations)
    # print(f"Shape of Weights: {weights.shape}")
    print("Model trained!")

    print("Training model using averaged perceptron...")
    averaged_weights = averaged_multiclass_perceptron(training_data, training_labels, iterations)
    # print(f"Shape of Averaged Weights: {averaged_weights.shape}")
    print("Model trained!")

    training_accuracy = compute_accuracy(weights, training_data, training_labels)
    testing_accuracy = compute_accuracy(weights, testing_data, testing_labels)

    averaged_training_accuracy = compute_accuracy(averaged_weights, training_data, training_labels)
    averaged_testing_accuracy = compute_accuracy(averaged_weights, testing_data, testing_labels)

    output = f"training-accuracy-standard-perceptron {training_accuracy:.4f} "
    output += f"testing-accuracy-standard-perceptron {testing_accuracy:.4f}\n"
    output += f"training-accuracy-averaged-perceptron {averaged_training_accuracy:.4f} "
    output += f"testing-accuracy-averaged-perceptron {averaged_testing_accuracy:.4f}"

    dump_output(output)

    print("Houston, we have output...")


if __name__ == "__main__":
    main()