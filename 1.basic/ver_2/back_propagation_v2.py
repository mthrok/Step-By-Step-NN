"""The multi-layer perceptron implementation with momentum. 
This file implements the multi-layer perceptron with ...
  one hidden layer,
  sigmoid function as activation in the hidden layer,
  softmax function as output function and 
  gradient descent with Nestrov's accelerated gradient.

The network is represented with four variables, W1, B1, W2 and B2,
corresponding weight matrices and bias vectors of the hidden layer and 
the output layer. 

When this file is run as script, MLP is trained on MNIST dataset, and 
the result is saved to a file. For the detail of command-line paramters, 
use call this file with -h option. 
"""
import sys, time, os, argparse
import numpy as np


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def softmax(x):
    y = np.exp(x)
    return y / np.sum(y)


def train_network(X_train, T_train, X_test, T_test, 
                  W1, B1, W2, B2, dW1, dB1, dW2, dB2, lr, mc, n_epochs):
    """
    Multiclass classification with NN
    Args:
      X_train(NumPy Array) : Data matrix (shape: n_train, n_input)
      T_train(NumPy Array) : Label matrix (shape: n_train, n_output)

      X_test(NumPy Array) : Data matrix (shape: n_test, n_input)
      T_test(NumPy Array) : Label matrix (shape: n_test, n_output)

      W1(NumPy Array) : Weight matrix in layer1 (shape: n_hidden, n_input)
      B1(NumPy Array) : Bias vector in layer1 (shape: n_hidden, 1)
      W2(NumPy Array) : Weight matrix in layer2 (shape: n_output, n_hidden)
      B2(NumPy Array) : Bias vector in layer2 (shape: n_output, 1)

      dW1(NumPy Array) : Previous channge of W1 (shape: n_hidden, n_input)
      dB1(NumPy Array) : Previous channge of B1 (shape: n_hidden, 1)
      dW2(NumPy Array) : Previous channge of B2 (shape: n_output, n_hidden)
      dB2(NumPy Array) : Previous channge of B2 (shape: n_output, 1)

      lr(float)     : Learning rate
      mc(float)     : Momentum coefficient
      n_epochs(int) : Maximum epochs to update the parameters

      where 
        n_train, n_test, are the number of training/test samples respertively. 
        n_input, n_output are the numbere of input/output features respectively
        n_hidden is the number of unit in the hidden layer

    Returns:
      W1, B1, W2, B2 (NumPy Array): Updated weight matrices and vectors
      ce_train(list of float)  : The history of cross entropy on training data.
      acc_train(list of float) : The history of cross entropy on training data.
      ce_test(list of float)   : The history of accuracy on training data.
      acc_test(list of float)  : The history of accuracy on training data.
    """
    # Get the dimension of input variables
    n_train, n_test = X_train.shape[0], X_test.shape[0]
    n_input, n_output = X_train.shape[1], T_train.shape[1]

    # Train the network
    ce_train, acc_train, ce_test, acc_test = [np.inf], [np.inf], [np.inf], [np.inf]
    try:
        for epoch in range(1, n_epochs+1):
            start_time = time.time()
            # Train on training data
            ce1, acc1 = 0.0, 0.0
            dEdW1, dEdB1 = np.zeros(W1.shape), np.zeros(B1.shape)
            dEdW2, dEdB2 = np.zeros(W2.shape), np.zeros(B2.shape)
            # Store the current weight, and jump with current gradient
            W1_0, B1_0, W2_0, B2_0 = W1, B1, W2, B2
            W1, B1, W2, B2 = W1+mc*dW1, B1+mc*dB1, W2+mc*dW2, B2+mc*dB2
            for i, (x, t) in enumerate(zip(X_train, T_train), start=1):
                print("Training {:>6}/{:<6}\r".format(i, n_train), end="")
                x_hid = x.reshape((-1, 1))
                ### Forward pass
                # Output from the hidden layer
                x_out = sigmoid(B1+np.dot(W1, x_hid))
                # Output from the output layer
                y_out = softmax(B2+np.dot(W2, x_out))

                ### Error Check
                t = t.reshape((-1, 1))
                # Cross-entropy
                ce1 += -np.sum(t*np.log(y_out)) / n_train
                # Accuracy
                if np.max(y_out)==np.max(y_out*t):
                    acc1 += 1 / n_train

                ### Backward pass
                # Output layer
                error = y_out - t
                dEdW2 += np.dot(error, x_out.T) / n_train
                dEdB2 += error / n_train
                # Hidden layer
                error = np.dot(W2.T, error) * x_out * (1-x_out)
                dEdW1 += np.dot(error, x_hid.T) / n_train
                dEdB1 += error / n_train

            # Update Weights
            dW1 = - lr * dEdW1 + mc * dW1
            dB1 = - lr * dEdB1 + mc * dB1
            dW2 = - lr * dEdW2 + mc * dW2
            dB2 = - lr * dEdB2 + mc * dB2
            W1, B1, W2, B2 = W1_0+dW1, B1_0+dB1, W2_0+dW2, B2_0+dB2

            # Evaluate error on test data
            ce2, acc2 = 0.0, 0.0
            for i, (x, t) in enumerate(zip(X_test, T_test), start=1):
                print("Testing {:>6}/{:<6}\r".format(i, n_test), end="")
                x_hid = x.reshape((-1, 1))
                ### Forward pass
                # Output from the hidden layer
                x_out = sigmoid(B1+np.dot(W1, x_hid))
                # Output from the output layer
                y_out = softmax(B2+np.dot(W2, x_out))

                ### Error Check
                t = t.reshape((-1, 1))
                # Cross-entropy
                ce2 += -np.sum(t*np.log(y_out)) / n_test
                # Accuracy
                if np.max(y_out)==np.max(y_out*t):
                    acc2 += 1 / n_test

            # Store cross entropy and accuracy
            ce_train.append(ce1)
            acc_train.append(acc1)

            # Store cross entropy and accuracy
            ce_test.append(ce2)
            acc_test.append(acc2)

            time_elapsed = time.time() - start_time
            print(("Epoch {:4} ({:6.2f} sec)   " 
                   "CE_train {:12.5e}({:12.5e})   CE_test {:12.5e}({:12.5e})   "
                   "ACC_train {:<6.3}   ACC_test {:<6.3}").format( \
                    epoch, time_elapsed, 
                    ce_train[-1], ce_train[-1]-ce_train[-2],
                    ce_test[-1], ce_test[-1]-ce_test[-2], 
                    acc_train[-1], acc_test[-1]))
    except KeyboardInterrupt:
        pass

    return W1, B1, W2, B2, dW1, dB1, dW2, dB2, ce_train[1:], acc_train[1:], ce_test[1:], acc_test[1:]


def main():
    from sklearn import datasets
    from sklearn.preprocessing import label_binarize
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train NN for MNIST dataset.')
    parser.add_argument('-hu', '--hidden_units', type=int, default=20,
                        help='The number of hidden units.')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.6,
                        help='Learning Rate')
    parser.add_argument('-e', '--epochs', type=int, default=200,
                        help='The number of maximum epoch')
    parser.add_argument('-m', '--momentum_coefficient', type=float, default=0.7,
                        help='The momentum coefficient')
    parser.add_argument('-i', '--input-file-path', type=str, 
                        default=None)
    parser.add_argument('-o', '--output-file-path', type=str, 
                        default=time.strftime("bp_v2_%Y%m%d-%H%M%S")+".npz")
    args = parser.parse_args()

    input_path, output_path = args.input_file_path, args.output_file_path
    lr, mc = args.learning_rate, args.momentum_coefficient
    n_hid, n_epochs = args.hidden_units, args.epochs
    n_in, n_out = 28*28, 10

    # Load previous data
    read = False
    if input_path is not None:
        print("Loading data from {}".format(input_path))
        try:
            data = np.load(input_path)
            n_hid, lr, mc = data['n_hid'], data['lr'], data['mc']
            W1, B1, W2, B2 = data['W1'], data['B1'], data['W2'], data['B2']
            dW1, dB1, dW2, dB2 = data['dW1'], data['dB1'], data['dW2'], data['dB2']
            ce_train = data['ce_train'].tolist()
            ce_test = data['ce_test'].tolist()
            acc_train = data['acc_train'].tolist()
            acc_test = data['acc_test'].tolist()
            if n_epochs>len(ce_train):
                n_epochs = n_epochs-len(ce_train)
            read = True
            print("")
        except:
            print(" Failed to load from {}\n".format(input_path))

    print("Paramteres:\n Hidden Units:{}".format(n_hid))
    print(" Learning rate:{}".format(lr))
    print(" Momentum coefficient:", mc)
    print(" Epochs:{}".format(n_epochs))
    print(" Output:{}\n".format(output_path))

    # Initialize the weight
    if not read:
        ce_train, acc_train, ce_test,  acc_test  = [], [], [], []
        # Make sure weights contain both positive and negative values!
        # Otherwise it's gonna stuck at local minimum toooooooo soon.
        c = 0.5
        W1 = np.random.uniform(-c, c, size=(n_hid, n_in))
        B1 = np.random.uniform(-c, c, size=(n_hid, 1))
        W2 = np.random.uniform(-c, c, size=(n_out, n_hid))
        B2 = np.random.uniform(-c, c, size=(n_out, 1))
        # It is interesting that changing uniform distribution to normal
        # distribution makes optimization speed very slow...
        # It seems that smaller initial weight gives better performance 
        # improevment at the biginning.
        '''
        W1 = np.random.normal(size=(n_hid, n_in))
        B1 = np.random.normal(size=(n_hid, 1))
        W2 = np.random.normal(size=(n_out, n_hid))
        B2 = np.random.normal(size=(n_out, 1))
        '''
        dW1, dB1 = np.zeros(W1.shape), np.zeros(B1.shape)
        dW2, dB2 = np.zeros(W2.shape), np.zeros(B2.shape)

    # Load image/label data
    print("Loading training/label data \n")
    mnist = datasets.fetch_mldata('MNIST Original')
    X = mnist['data'].astype(np.float) / 255
    T = label_binarize(mnist['target'], classes=np.unique(mnist['target']))
    X_train, X_test = X[:60000, :], X[60000:, :]
    T_train, T_test = T[:60000, :], T[60000:, :]
    
    # Train
    W1, B1, W2, B2, dW1, dB1, dW2, dB2, \
        ce_train_new, acc_train_new, ce_test_new, acc_test_new = \
        train_network(X_train, T_train, X_test, T_test,
                      W1, B1, W2, B2, dW1, dB1, dW2, dB2, lr, mc, n_epochs)
    ce_train.extend(ce_train_new)
    acc_train.extend(acc_train_new)
    ce_test.extend(ce_test_new)
    acc_test.extend(acc_test_new)
    
    # Save
    np.savez(output_path, 
             W1=W1, B1=B1, W2=W2, B2=B2, dW1=dW1, dB1=dB1, dW2=dW2, dB2=dB2, 
             lr=lr, mc=mc, n_hid=n_hid, 
             ce_train=ce_train, acc_train=acc_train, 
             ce_test=ce_test, acc_test=acc_test)
    print("Saved to {}\n\n".format(output_path))


if __name__=="__main__":
    main()
