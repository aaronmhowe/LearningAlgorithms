import numpy as np
import os


# data
vocab = []
training_data = []
training_labels = []
testing_data = []
testing_labels = []
stopwords = set()

# set number of iterations 1 to 20
iterations = 20

# learning rate
a = 1

in_stoplist = os.path.join(os.path.dirname(__file__), 'fortune_cookie_data/stoplist.txt')
in_testdata = os.path.join(os.path.dirname(__file__), 'fortune_cookie_data/testdata.txt')
in_testlabels = os.path.join(os.path.dirname(__file__), 'fortune_cookie_data/testlabels.txt')
in_traindata = os.path.join(os.path.dirname(__file__), 'fortune_cookie_data/traindata.txt')
in_trainlabels = os.path.join(os.path.dirname(__file__), 'fortune_cookie_data/trainlabels.txt')


"""
Function for reading and loading file data into memory
"""
def read_files():

    try:
        global training_data, training_labels, testing_data, testing_labels, stopwords
        with open(in_stoplist, 'r') as f:
            stopwords = set(f.read().splitlines())
        with open(in_testdata, 'r') as f:
            testing_data = f.read().splitlines()
        with open(in_testlabels, 'r') as f:
            testing_labels = f.read().splitlines()
        with open(in_traindata, 'r') as f:
            training_data = f.read().splitlines()
        with open(in_trainlabels, 'r') as f:
            training_labels = [int(label) for label in f.read().splitlines()]

        # print statements specifically for debugging purposes
        print(f"Reading data from {len(training_data)} traindata.txt")
        print(f"Reading data from {len(testing_data)} testdata.txt")
        print(f"Reading data from {len(stopwords)} stopwords.txt")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Working Directory: {os.getcwd()}")
        print(f"Fortune Cookie Data: ")
        print(os.listdir(os.path.dirname(in_stoplist)))


"""
Conversion function that converts input data into a feature vector when called in function pre_process_data()
@param vocab_words: represents the set of messages being split into sets of words from training and testing data
@param hash: as used in pre_process_data(), this is a dictionary used to numerically index the sets of words
@return word_index: words indexed 1 to 20
"""
def conversion(vocab_words, hash):
    split_words = vocab_words.lower().split()
    word_index = [0] * len(vocab)

    for words in split_words:
        if words in hash:
            word_index[hash[words]] = 1

    return word_index


"""
Pre-processing step in the task of converting messages into features
"""
def pre_process_data():

    global vocab, training_data, testing_data
    vocabulary = set()

    # populating the set of vocabulary words
    for message in training_data:
        vocab_words = message.lower().split()
        vocabulary.update(vocab_word for vocab_word in vocab_words if vocab_word not in stopwords)

    # sorting in alphabetical order
    vocab = sorted(list(vocabulary))
    hash = {vocab_word: i for i, vocab_word in enumerate(vocab)}
    training_data = [conversion(message, hash) for message in training_data]
    testing_data = [conversion(message, hash) for message in testing_data]
    # debug print
    print(f"Feature vector defined with size: {len(vocab)}")


"""
Function that implements the standard perceptron algorithm
@param data: input data
@param labels: true label (1 or -1)
@param itr: number of iterations (maximum of 20)
@return w: weight vector
"""
def perceptron(data, labels, itr):

    features = len(data[0])
    # w = [0, 0, 0, 0, 0, 0]
    w = np.zeros(features)

    for i in range(itr):
        mistake_count = 0

        for x, y in zip(data, labels):
            x = np.array(x)
            y_prediction = np.sign(np.dot(w, x))

            # update rule
            if y_prediction != y:
                w += y * x
                mistake_count += 1

        training_accuracy = compute_accuracy(w, data, labels)
        testing_accuracy = compute_accuracy(w, testing_data, testing_labels)

        # debugging
        print(f"{mistake_count} mistakes on {itr} iterations with {training_accuracy} training accuracy and {testing_accuracy} testing_accuracy.")

        if mistake_count == 0:
            break

    return w


"""
Function for computing the training and testing accuracy of the learning model
@param w:
@param data:
@param labels:
@return training and testing accuracy
"""
def compute_accuracy(w, data, labels):
    accuracy = sum(1 for x, y in zip(data, labels) if np.sign(np.dot(w, x)) == y)
    return accuracy / len(data)


"""
Function that implements averaged perceptron
@param data: input data
@param labels: true label
@param itr: max number of iterations @ 20
@return w_avg: vector of averaged weights
"""
def averaged_perceptron(data, labels, itr):

    features = len(data[0])
    w = np.zeros(features)
    # vector of averaged weights
    w_avg = np.zeros(features)
    # tracking number of weight updates
    update_count = 1

    for i in range(itr):
        mistake_count = 0

        for x, y in zip(data, labels):
            x = np.array(x)
            y_prediction = np.sign(np.dot(w, x))

            # update rule
            if y_prediction != y:
                w += y * x
                mistake_count += 1
            # computing averaged perceptron
            w_avg += update_count * w
            update_count += 1
                
        training_accuracy = compute_accuracy(w_avg, data, labels)
        testing_accuracy = compute_accuracy(w_avg, testing_data, testing_labels)

        # debugging
        print(f"{mistake_count} mistakes on {itr + 1} iterations, with {training_accuracy:.4f} training accuracy and {testing_accuracy:.4f} testing accuracy.")

        # repeat until convergence or maximum iterations reached
        if mistake_count == 0:
            break

    w_avg /= update_count
    return w_avg


"""
Function to make predictions on training examples
@param w: weight vector
@param feature: feature vector
@param return: class 1 or class 0
"""
def prediction(w, feature):
    predict = np.dot(w, feature)
    if predict > 0:
        return 1
    else:
        return 0


"""
Simple function for dumping program output
@param file: the output file to dump into
@param output: the output to be dumped into the output file
"""
def dump_output(file, output):

    with open(file, 'w') as f:
        f.write(output)


"""
Main Application Function
"""
def main():


    read_files()

    pre_process_data()

    print("Training the model using the perceptron algorithm...")
    weights = perceptron(training_data, training_labels, iterations)
    print("Model trained using perceptron!")

    training_accuracy = compute_accuracy(weights, training_data, training_labels)
    testing_accuracy = compute_accuracy(weights, testing_data, testing_labels)

    print("Training the model using averaged perceptron...")
    averaged_weights = averaged_perceptron(training_data, training_labels, iterations)
    print("Model trained using averaged perceptron!")

    averaged_training_accuracy = compute_accuracy(averaged_weights, training_data, training_labels)
    averaged_testing_accuracy = compute_accuracy(averaged_weights, testing_data, testing_labels)

    print("Computing Fortune Cookie Perceptron...")
    print(f"Averaged Training Accuracy: {averaged_training_accuracy}, Averaged Testing Accuracy: {averaged_testing_accuracy}")

    output = (f"Fortune Cookie Classifier Learning:\n"
              f"\n"
              f"Standard Perceptron:\n"
              f"Training Accuracy: {training_accuracy:.4f}\n"
              f"Testing Accuracy: {testing_accuracy:.4f}\n"
              f"Averaged Perceptron:\n"
              f"Averaged Training Accuracy: {averaged_training_accuracy:.4f}\n"
              f"Averaged Testing Accuracy: {averaged_testing_accuracy:.4f}\n"
              f"\n"
              f"OCR MultiClassifier Learning:\n"
              f"\n")
    
    dump_output('output.txt', output)
    print("Houston, we have output...")

if __name__ == "__main__":
    main()