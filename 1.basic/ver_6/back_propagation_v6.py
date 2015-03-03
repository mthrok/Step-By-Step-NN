"""The multi-layer perceptron implementation with momentum. 
This file implements the multi-layer perceptron with ...
  two hidden layers,
  ReLU function as activation in the hidden layer,
  softmax function as output function, 
  gradient descent with Nestrov's accelerated gradient and 
  mini-batch updates.

The network is represented with four variables, W1, B1, W2 and B2,
corresponding weight matrices and bias vectors of the hidden layer and 
the output layer. 

When this file is run as script, MLP is trained on MNIST dataset, and 
the result is saved to a file. For the detail of command-line paramters, 
use call this file with -h option. 
"""
import sys, time, os, argparse
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle


def dReL(x):
    return (np.sign(x)+1) / 2


def ReL(x):
    return np.maximum(0, x)


def softmax(x):
    y = np.exp(x)
    return y / np.sum(y)


def train_network(X_train, T_train, X_valid, T_valid, X_test, T_test, 
                  W1, B1, W2, B2, W3, B3, dW1, dB1, dW2, dB2, dW3, dB3, 
                  lr, mc, n_batches, n_epochs):
    """
    Multiclass classification with NN
    Args:
      X_train(NumPy Array) : 
        Data matrix for training (shape: n_train, n_input)
      T_train(NumPy Array) : 
        Label matrix for training (shape: n_train, n_output)

      X_valid(NumPy Array) : 
        Data matrix for validation (shape: n_valid, n_input)
      T_valid(NumPy Array) : 
        Label matrix for validation (shape: n_valid, n_output)

      X_test(NumPy Array) : Data matrix for test (shape: n_test, n_input)
      T_test(NumPy Array) : Label matrix for test (shape: n_test, n_output)

      W1(NumPy Array) : Weight matrix in layer1 (shape: n_hidden, n_input)
      B1(NumPy Array) : Bias vector in layer1 (shape: n_hidden1, 1)
      W2(NumPy Array) : Weight matrix in layer2 (shape: n_hidden2, n_hidden1)
      B2(NumPy Array) : Bias vector in layer2 (shape: n_hidden2, 1)
      W3(NumPy Array) : Weight matrix in layer3 (shape: n_output, n_hidden2)
      B3(NumPy Array) : Bias vector in layer3 (shape: n_output, 1)

      dW1(NumPy Array) : Previous channge of W1 (shape: n_hidden1, n_input)
      dB1(NumPy Array) : Previous channge of B1 (shape: n_hidden1, 1)
      dW2(NumPy Array) : Previous channge of W2 (shape: n_hidden2, n_hidden1)
      dB2(NumPy Array) : Previous channge of B2 (shape: n_hidden2, 1)
      dW3(NumPy Array) : Previous channge of W3 (shape: n_output, n_hidden2)
      dB3(NumPy Array) : Previous channge of B3 (shape: n_output, 1)

      lr(float)      : Learning rate
      mc(float)      : Momentum coefficient
      n_batches(int) : The number of batches. In each epoch, the training data 
        is shuffled and devided into this number of mini-batches
      n_epochs(int)  : Maximum epochs to update the parameters

      where 
        n_train, n_valid and n_test are the number of training/validatoin/test 
          samples respertively. 
        n_input, n_output are the numbere of input/output features respectively
        n_hidden1 and n_hidden2 are the numbers of unit in the hidden layer1 and 
          2 respectively

    Returns:
      W1, B1, W2, B2, W3, B3 (NumPy Array): Updated weight matrices and vectors
      ce_train(list of float)  : The history of cross entropy on training data.
      acc_train(list of float) : The history of cross entropy on training data.
      ce_test(list of float)   : The history of accuracy on training data.
      acc_test(list of float)  : The history of accuracy on training data.
    """
    # Get the dimension of input variables
    n_valid, n_test = X_valid.shape[0], X_test.shape[0]
    n_input, n_output = X_train.shape[1], T_train.shape[1]

    # Train the network
    ce_train, acc_train, ce_test, acc_test = \
        [np.inf], [np.inf], [np.inf], [np.inf]
    try:
        for epoch in range(1, n_epochs+1):
            start_time = time.time()
            # Train on each mini-batch
            for batch in range(n_batches):
                X_batch, T_batch = \
                    X_train[batch::n_batches], T_train[batch::n_batches]
                # Initialize gradient
                dEdW1, dEdB1 = np.zeros(W1.shape), np.zeros(B1.shape)
                dEdW2, dEdB2 = np.zeros(W2.shape), np.zeros(B2.shape)
                dEdW3, dEdB3 = np.zeros(W3.shape), np.zeros(B3.shape)
                # Store the current weight, and jump with current gradient
                W1_0, B1_0, W2_0, B2_0, W3_0, B3_0 = W1, B1, W2, B2, W3, B3
                W1, B1 = W1+mc*dW1, B1+mc*dB1 
                W2, B2 = W2+mc*dW2, B2+mc*dB2
                W3, B3 = W3+mc*dW3, B3+mc*dB3
                n_train = X_batch.shape[0]
                for i, (x, t) in enumerate(zip(X_batch, T_batch), start=1):
                    print(("Training: Batch{:>3}/{}, Sample{:>6}/{}\r"
                           "").format(batch, n_batches, i, n_train), end="")
                    x_hid1 = x.reshape((-1, 1))
                    ### Forward pass
                    # Hidden layer1
                    x_hid2 = ReL(B1+np.dot(W1, x_hid1))
                    # Hidden layer2
                    x_out = ReL(B2+np.dot(W2, x_hid2))
                    # Output layer
                    y_out = softmax(B3+np.dot(W3, x_out))

                    ### Backward pass
                    t = t.reshape((-1, 1))
                    # Output layer
                    error = y_out - t
                    dEdW3 += np.dot(error, x_out.T) / n_train
                    dEdB3 += error / n_train
                    # Hidden layer2
                    error = np.dot(W3.T, error) * dReL(B2+np.dot(W2, x_hid2))
                    dEdW2 += np.dot(error, x_hid2.T) / n_train
                    dEdB2 += error / n_train
                    # Hidden layer1
                    error = np.dot(W2.T, error) * dReL(B1+np.dot(W1, x_hid1))
                    dEdW1 += np.dot(error, x_hid1.T) / n_train
                    dEdB1 += error / n_train

                # Update Weights
                dW1 = - lr * dEdW1 + mc * dW1
                dB1 = - lr * dEdB1 + mc * dB1
                dW2 = - lr * dEdW2 + mc * dW2
                dB2 = - lr * dEdB2 + mc * dB2
                dW3 = - lr * dEdW3 + mc * dW3
                dB3 = - lr * dEdB3 + mc * dB3
                W1, B1 = W1_0+dW1, B1_0+dB1
                W2, B2 = W2_0+dW2, B2_0+dB2
                W3, B3 = W3_0+dW3, B3_0+dB3

            # Evaluate error on validation data
            ce1, acc1 = 0.0, 0.0
            for i, (x, t) in enumerate(zip(X_valid, T_valid), start=1):
                print("Validating {:>6}/{}       \r".format(i, n_valid), end="")
                x_hid1 = x.reshape((-1, 1))
                ### Forward pass
                # Hidden layer1
                x_hid2 = ReL(B1+np.dot(W1, x_hid1))
                # Hidden layer2
                x_out = ReL(B2+np.dot(W2, x_hid2))
                # Output layer
                y_out = softmax(B3+np.dot(W3, x_out))

                ### Error Check
                t = t.reshape((-1, 1))
                # Cross-entropy
                ce1 += -np.sum(t*np.log(y_out)) / n_valid
                # Accuracy
                if np.max(y_out)==np.max(y_out*t):
                    acc1 += 1 / n_valid

            # Evaluate error on test data
            ce2, acc2 = 0.0, 0.0
            for i, (x, t) in enumerate(zip(X_test, T_test), start=1):
                print("Testing {:>6}/{}           \r".format(i, n_test), end="")
                x_hid1 = x.reshape((-1, 1))
                ### Forward pass
                # Hidden layer1
                x_hid2 = ReL(B1+np.dot(W1, x_hid1))
                # Hidden layer2
                x_out = ReL(B2+np.dot(W2, x_hid2))
                # Output layer
                y_out = softmax(B3+np.dot(W3, x_out))

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
            print(("Epoch {:4} ({:6.2f} sec) " 
                   "CE_valid {:12.5e}({:12.5e}) CE_test {:12.5e}({:12.5e}) "
                   "ACC_valid {:<6.3} ACC_test {:<6.3} "
                   "|W1| {:>6.5e} |W2| {:>6.5e} |W3| {:>6.5e}").format( \
                    epoch, time_elapsed, 
                    ce_train[-1], ce_train[-1]-ce_train[-2],
                    ce_test[-1], ce_test[-1]-ce_test[-2], 
                    acc_train[-1], acc_test[-1], 
                    np.sum(np.square(W1)), np.sum(np.square(W2)), 
                    np.sum(snp.square(W3)) ))
    except KeyboardInterrupt:
        pass

    return W1, B1, W2, B2, W3, B3, dW1, dB1, dW2, dB2, dW3, dB3, \
        ce_train[1:], acc_train[1:], ce_test[1:], acc_test[1:]


def main():
    from sklearn import datasets
    from sklearn.preprocessing import label_binarize

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train NN for MNIST dataset.')
    parser.add_argument('-hu', '--hidden_units', type=int, nargs=2, 
                        default=[11, 21], help='The number of hidden units.')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.6,
                        help='Learning Rate')
    parser.add_argument('-e', '--epochs', type=int, default=200,
                        help='The number of maximum epoch')
    parser.add_argument('-m', '--momentum_coefficient', type=float, default=0.7,
                        help='The momentum coefficient')
    parser.add_argument('-b', '--batches', type=int, default=10,
                        help='The number of mini-batches')
    parser.add_argument('-i', '--input-file-path', type=str, 
                        default=None)
    parser.add_argument('-o', '--output-file-path', type=str, 
                        default=time.strftime("bp_v5_%Y%m%d-%H%M%S")+".npz")
    args = parser.parse_args()

    input_path, output_path = args.input_file_path, args.output_file_path
    lr, mc, nb = args.learning_rate, args.momentum_coefficient, args.batches
    n_hid, n_epochs = args.hidden_units, args.epochs
    n_in, n_out = 28*28, 10

    # Load previous data
    read = False
    if input_path is not None:
        print("Loading data from {}".format(input_path))
        try:
            data = np.load(input_path)
            n_hid, lr, mc, nb = data['n_hid'], data['lr'], data['mc'], data['nb']
            W1, B1 = data['W1'], data['B1'] 
            W2, B2 = data['W2'], data['B2']
            W3, B3 = data['W3'], data['B3']
            dW1, dB1 = data['dW1'], data['dB1']
            dW2, dB2 = data['dW2'], data['dB2']
            dW3, dB3 = data['dW3'], data['dB3']
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

    print("Paramteres:\n Hidden Units:{}, {}".format(n_hid[0], n_hid[1]))
    print(" Learning rate:{}".format(lr))
    print(" Momentum coefficient:", mc)
    print(" Epochs:{}".format(n_epochs))
    print(" Number of batches:{}".format(nb))
    print(" Output:{}\n".format(output_path))

    # Initialize the weight
    if not read:
        ce_train, acc_train, ce_test,  acc_test  = [], [], [], []
        # Make sure weights contain both positive and negative values!
        # Otherwise it's gonna stuck at local minimum toooooooo soon.
        c = 0.5
        W1 = np.random.uniform(-c, c, size=(n_hid[0], n_in))
        B1 = np.random.uniform(-c, c, size=(n_hid[0], 1))
        W2 = np.random.uniform(-c, c, size=(n_hid[1], n_hid[0]))
        B2 = np.random.uniform(-c, c, size=(n_hid[1], 1))
        W3 = np.random.uniform(-c, c, size=(n_out, n_hid[1]))
        B3 = np.random.uniform(-c, c, size=(n_out, 1))
        dW1, dB1 = np.zeros(W1.shape), np.zeros(B1.shape)
        dW2, dB2 = np.zeros(W2.shape), np.zeros(B2.shape)
        dW3, dB3 = np.zeros(W3.shape), np.zeros(B3.shape)

    # Load image/label data
    print("Loading training/label data \n")
    mnist = datasets.fetch_mldata('MNIST Original')
    X = mnist['data'].astype(np.float) / 255
    T = label_binarize(mnist['target'], classes=np.unique(mnist['target']))
    # Split into traint/validation/test set
    ind_train = [i+c*6000 for c in range(10) for i in range(5000)]
    ind_valid = [i+c*6000 for c in range(10) for i in range(5000, 6000)]
    X_train, X_valid, X_test = X[ind_train, :], X[ind_valid, :], X[60000:, :]
    T_train, T_valid, T_test = T[ind_train, :], T[ind_valid, :], T[60000:, :] 

    # Train
    W1, B1, W2, B2, W3, B3, dW1, dB1, dW2, dB2, dW3, dB3, \
        ce_train_new, acc_train_new, ce_test_new, acc_test_new = \
        train_network(X_train, T_train, X_valid, T_valid, X_test, T_test,
                      W1, B1, W2, B2, W3, B3, 
                      dW1, dB1, dW2, dB2, dW3, dB3, 
                      lr, mc, nb, n_epochs)
    ce_train.extend(ce_train_new)
    acc_train.extend(acc_train_new)
    ce_test.extend(ce_test_new)
    acc_test.extend(acc_test_new)
    
    # Save
    np.savez(output_path, 
             W1=W1, B1=B1, W2=W2, B2=B2, W3=W3, B3=B3,  
             dW1=dW1, dB1=dB1, dW2=dW2, dB2=dB2, dW3=dW3, dB3=dB3, 
             lr=lr, mc=mc, nb=nb, n_hid=n_hid, 
             ce_train=ce_train, acc_train=acc_train, 
             ce_test=ce_test, acc_test=acc_test)
    print("Saved to {}\n\n".format(output_path))


if __name__=="__main__":
    main()
