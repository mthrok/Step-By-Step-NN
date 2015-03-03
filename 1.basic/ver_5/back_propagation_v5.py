"""The multi-layer perceptron implementation with momentum. 
This file implements the multi-layer perceptron with ...
  one hidden layer,
  tanh function as activation in the hidden layer,
  softmax function as output function, 
  gradient descent with Nestrov's accelerated gradient,
  weight decay and mini-batch updates.

The network is represented with four variables, W1, B1, W2 and B2,
corresponding weight matrices and bias vectors of the hidden layer and 
the output layer. 

When this file is run as script, MLP is trained on MNIST dataset, and 
the result is saved to a file. For the detail of command-line paramters, 
use call this file with -h option. 
"""
import sys, time, os, argparse
import numpy as np
from numpy import tanh
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle


def softmax(x):
    y = np.exp(x)
    return y / np.sum(y)


def train_network(X_train, T_train, X_valid, T_valid, X_test, T_test, W1, B1, 
                  W2, B2, dW1, dB1, dW2, dB2, lr, mc, wd, n_batches, n_epochs):
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
      B1(NumPy Array) : Bias vector in layer1 (shape: n_hidden, 1)
      W2(NumPy Array) : Weight matrix in layer2 (shape: n_output, n_hidden)
      B2(NumPy Array) : Bias vector in layer2 (shape: n_output, 1)

      dW1(NumPy Array) : Previous channge of W1 (shape: n_hidden, n_input)
      dB1(NumPy Array) : Previous channge of B1 (shape: n_hidden, 1)
      dW2(NumPy Array) : Previous channge of B2 (shape: n_output, n_hidden)
      dB2(NumPy Array) : Previous channge of B2 (shape: n_output, 1)

      lr(float)      : Learning rate
      mc(float)      : Momentum coefficient
      wd(float)      : Weight decay
      n_batches(int) : The number of batches. In each epoch, the training data 
        is shuffled and devided into this number of mini-batches
      n_epochs(int)  : Maximum epochs to update the parameters

      where 
        n_train, n_valid and n_test are the number of training/validatoin/test 
        samples respertively. 
        n_input, n_output are the numbere of input/output features respectively
        n_hidden is the number of unit in the hidden layer

    Returns:
      W1, B1, W2, B2 (NumPy Array): Updated weight matrices and vectors
      ce_valid(list of float)  : The history of cross entropy on training data.
      acc_valid(list of float) : The history of cross entropy on training data.
      ce_test(list of float)   : The history of accuracy on training data.
      acc_test(list of float)  : The history of accuracy on training data.
      size_W1(list of float)   : The history of the size of W1
      size_W2(list of float)   : The history of the size of W2
    """
    # Get the dimension of input variables
    n_valid, n_test = X_valid.shape[0], X_test.shape[0]
    n_input, n_hid, n_output = X_train.shape[1], W1.shape[0], T_train.shape[1]

    # Train the network
    ce_valid, acc_valid, ce_test, acc_test, size_W1, size_W2 = \
        [np.inf], [np.inf], [np.inf], [np.inf], [], []
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
                # Store the current weight, and jump with current gradient
                W1_0, B1_0, W2_0, B2_0 = W1, B1, W2, B2
                W1, B1, W2, B2 = W1+mc*dW1, B1+mc*dB1, W2+mc*dW2, B2+mc*dB2
                n_train = X_batch.shape[0]
                for i, (x, t) in enumerate(zip(X_batch, T_batch), start=1):
                    print(("Training: Batch{:>3}/{}, Sample{:>6}/{}\r"
                           "").format(batch, n_batches, i, n_train), end="")
                    x_hid = x.reshape((-1, 1))
                    ### Forward pass
                    # Output from the hidden layer
                    x_out = tanh(B1+np.dot(W1, x_hid))
                    # Output from the output layer
                    y_out = softmax(B2+np.dot(W2, x_out))

                    ### Backward pass
                    t = t.reshape((-1, 1))
                    # Output layer
                    error = y_out - t
                    dEdW2 += np.dot(error, x_out.T) / n_train
                    dEdB2 += error / n_train
                    # Hidden layer
                    error = np.dot(W2.T, error) * (1 - x_out * x_out)
                    dEdW1 += np.dot(error, x_hid.T) / n_train
                    dEdB1 += error / n_train
                # Update Weights
                dW1 = - lr * dEdW1 -lr * wd * W1 + mc * dW1
                dB1 = - lr * dEdB1 -lr * wd * B1 + mc * dB1
                dW2 = - lr * dEdW2 -lr * wd * W2 + mc * dW2
                dB2 = - lr * dEdB2 -lr * wd * B2 + mc * dB2
                W1, B1, W2, B2 = W1_0+dW1, B1_0+dB1, W2_0+dW2, B2_0+dB2

            # Evaluate error on validation data
            ce1, acc1 = 0.0, 0.0
            for i, (x, t) in enumerate(zip(X_valid, T_valid), start=1):
                print("Validating {:>6}/{}       \r".format(i, n_valid), end="")
                x_hid = x.reshape((-1, 1))
                ### Forward pass
                # Output from the hidden layer
                x_out = tanh(B1+np.dot(W1, x_hid))
                # Output from the output layer
                y_out = softmax(B2+np.dot(W2, x_out))

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
                x_hid = x.reshape((-1, 1))
                ### Forward pass
                # Output from the hidden layer
                x_out = tanh(B1+np.dot(W1, x_hid))
                # Output from the output layer
                y_out = softmax(B2+np.dot(W2, x_out))

                ### Error Check
                t = t.reshape((-1, 1))
                # Cross-entropy
                ce2 += -np.sum(t*np.log(y_out)) / n_test
                # Accuracy
                if np.max(y_out)==np.max(y_out*t):
                    acc2 += 1 / n_test

            # Store cross entropy and accuracy for training
            ce_valid.append(ce1)
            acc_valid.append(acc1)

            # Store cross entropy and accuracy for test
            ce_test.append(ce2)
            acc_test.append(acc2)
            
            # Stor the sizes
            size_W1.append(np.sum(np.square(W1)))
            size_W2.append(np.sum(np.square(W2)))

            time_elapsed = time.time() - start_time
            if (epoch % 30 == 1):
                print(("{:<5} {:>6} "
                       "{:<12}{:>14} {:<12}{:>14} "
                       "AC {:<6} {:<6} "
                       "{:>12} {:>12}").format(\
                        "Epoch", "Time", 
                        "CE_VLD", " ", "CE_TST", " ", 
                        "VLD", "TST", 
                        "|W1|", "|W2|"))
            print(("{:5} {:6.2f} "
                   "{:12.5e}({:12.5e}) {:12.5e}({:12.5e}) "
                   "   {:<6.3} {:<6.3} "
                   "{:>12.5e} {:>12.5e}").format( \
                    epoch, time_elapsed, 
                    ce_valid[-1], ce_valid[-1]-ce_valid[-2],
                    ce_test[-1], ce_test[-1]-ce_test[-2], 
                    acc_valid[-1], acc_test[-1],
                    size_W1[-1], size_W2[-1]))
    except KeyboardInterrupt:
        pass

    return W1, B1, W2, B2, dW1, dB1, dW2, dB2, \
        ce_valid[1:], acc_valid[1:], ce_test[1:], acc_test[1:], \
        size_W1, size_W2


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
    parser.add_argument('-b', '--batches', type=int, default=10,
                        help='The number of mini-batches')
    parser.add_argument('-w', '--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('-i', '--input-file-path', type=str, 
                        default=None)
    parser.add_argument('-o', '--output-file-path', type=str, 
                        default=time.strftime("bp_v5_%Y%m%d-%H%M%S")+".npz")
    args = parser.parse_args()

    input_path, output_path = args.input_file_path, args.output_file_path
    lr, mc = args.learning_rate, args.momentum_coefficient
    nb, wd = args.batches, args.weight_decay
    n_hid, n_epochs = args.hidden_units, args.epochs
    n_in, n_out = 28*28, 10

    # Load previous data
    read = False
    if input_path is not None:
        print("Loading data from {}".format(input_path))
        try:
            data = np.load(input_path)
            n_hid, lr, mc = data['n_hid'], data['lr'], data['mc']
            nb, wd = data['nb'], data['wd']
            W1, B1, W2, B2 = data['W1'], data['B1'], data['W2'], data['B2']
            dW1, dB1, dW2, dB2 = data['dW1'], data['dB1'], data['dW2'], data['dB2']
            ce_valid = data['ce_valid'].tolist()
            ce_test = data['ce_test'].tolist()
            acc_valid = data['acc_valid'].tolist()
            acc_test = data['acc_test'].tolist()
            size_W1 = data['size_W1'].tolist()
            size_W2 = data['size_W2'].tolist()
            if n_epochs>len(ce_valid):
                n_epochs = n_epochs-len(ce_valid)
            read = True
            print("")
        except:
            print(" Failed to load from {}\n".format(input_path))

    print("Paramteres:\n Hidden Units:{}".format(n_hid))
    print(" Learning rate:{}".format(lr))
    print(" Momentum coefficient:", mc)
    print(" Weight decay:", wd)
    print(" Epochs:{}".format(n_epochs))
    print(" Number of batches:{}".format(nb))
    print(" Output:{}\n".format(output_path))

    # Initialize the weight
    if not read:
        ce_valid, acc_valid, ce_test, acc_test, size_W1, size_W2\
            = [], [], [], [], [], []
        # Make sure weights contain both positive and negative values!
        # Otherwise it's gonna stuck at local minimum toooooooo soon.
        c = 0.5
        W1 = np.random.uniform(-c, c, size=(n_hid, n_in))
        B1 = np.random.uniform(-c, c, size=(n_hid, 1))
        W2 = np.random.uniform(-c, c, size=(n_out, n_hid))
        B2 = np.random.uniform(-c, c, size=(n_out, 1))
        dW1, dB1 = np.zeros(W1.shape), np.zeros(B1.shape)
        dW2, dB2 = np.zeros(W2.shape), np.zeros(B2.shape)

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
    W1, B1, W2, B2, dW1, dB1, dW2, dB2, \
        ce_valid_new, acc_valid_new, ce_test_new, acc_test_new, \
        size_W1_new, size_W2_new = \
        train_network(X_train, T_train, X_valid, T_valid, X_test, T_test,
                      W1, B1, W2, B2, dW1, dB1, dW2, dB2, lr, mc, wd, nb, n_epochs)
    ce_valid.extend(ce_valid_new)
    acc_valid.extend(acc_valid_new)
    ce_test.extend(ce_test_new)
    acc_test.extend(acc_test_new)
    size_W1.extend(size_W1_new)
    size_W2.extend(size_W2_new)

    # Save
    np.savez(output_path, 
             W1=W1, B1=B1, W2=W2, B2=B2, dW1=dW1, dB1=dB1, dW2=dW2, dB2=dB2, 
             lr=lr, mc=mc, wd=wd, nb=nb, n_hid=n_hid, 
             ce_valid=ce_valid, acc_valid=acc_valid, 
             ce_test=ce_test, acc_test=acc_test, 
             size_W1=size_W1, size_W2=size_W2)
    print("Saved to {}\n\n".format(output_path))


if __name__=="__main__":
    main()
