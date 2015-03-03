import sys
import numpy as np
import matplotlib.pyplot as plt


def normalize(img):
    return img / np.max(img)


def main():
    if len(sys.argv)<2:
        print("Usage: {} input_npz_file".format(sys.argv[0]))
        sys.exit(1)

    for input_file in sys.argv[1:]:
        try:
            data = np.load(input_file)
            print("Loaded", input_file)
        except:
            print("Failed to load", input_file)
            continue

        key_to_plot = 'W1'

        n_row = np.ceil(np.sqrt(data[key_to_plot].shape[0]))    
        fig = plt.figure()
        print("Plotting")
        for i, w in enumerate(data[key_to_plot], start=1):
            ax = fig.add_subplot(n_row, n_row, i)
            img = normalize(w.reshape(28, 28))
            ax.imshow(img, cmap = plt.get_cmap('gray'))
    print("Plot ready.")
    plt.show()


if __name__=="__main__":
    main()
