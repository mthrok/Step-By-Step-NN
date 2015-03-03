"""Display the content of npz file"""
import numpy as np
import os, sys

def main():
    if len(sys.argv) < 2:
        print("Usage: {} *input_npz_files".format(sys.argv[0]))
    for argv in sys.argv[1:]:
        try:
            data = np.load(argv)
            print("Loaded {}".format(argv))
            for key in data.keys():
                print(" {}, shape ".format(key), data[key].shape)
            print("")
        except:
            print("Failed to load {}".format(argv))

if __name__=="__main__":
    main()
