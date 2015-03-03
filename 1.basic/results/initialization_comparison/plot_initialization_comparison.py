import os
import numpy as np
import matplotlib.pyplot as plt


def vary_colors(N):
    colors = plt.get_cmap('jet')(np.linspace(0, 1.0, N))
    for i in range(len(colors)):
        colors[i, 0:3] /= np.max(colors[i, 0:3])
    return colors


def plot_comparison(data_set1, data_set2, data_set3, data_set4, out_dir):
    plt.figure()

    # Plot each data 
    for data1, data2, data3, data4 in zip(data_set1, data_set2, data_set3, data_set4):
        lr = np.asscalar(data1['lr'])
        n_hid = np.asscalar(data1['n_hid'])
        # Set up figure
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross Entropy")
        ax1.set_title(\
            "Cross Entropy for $\\alpha$:{}, $N_{{hid}}:${}".format(lr, n_hid))
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title(\
            "Accuracy for $\\alpha$:{}, $N_{{hid}}:${}".format(lr, n_hid))
        min_epochs = np.inf

        # Generate colors
        clrs = vary_colors(4)
        # Plot data1
        lbl = "Initialized with normal \ndistribution, "
        n_epochs = len(data1['ce_train'])
        epochs_train = [ep for ep in range(n_epochs)]
        epochs_test  = [ep+1 for ep in range(n_epochs)]
        ax1.plot(epochs_train, data1['ce_train'], '-', color=clrs[0], 
                 label=lbl+"$Train$")
        ax1.plot(epochs_test, data1['ce_test'], '--', color=clrs[0], 
                 label=lbl+"$Test$")
        ax2.plot(epochs_train, data1['acc_train'], '-', color=clrs[0])
        ax2.plot(epochs_test, data1['acc_test'], '--', color=clrs[0])
        min_epochs = n_epochs if min_epochs>n_epochs else min_epochs

        # Plot data2
        lbl = "Initialized with \nuniform distribution \n range [-0.5, 0.5]\n, "
        n_epochs = len(data2['ce_train'])
        epochs_train = [ep for ep in range(n_epochs)]
        epochs_test  = [ep+1 for ep in range(n_epochs)]
        ax1.plot(epochs_train, data2['ce_train'], '-', color=clrs[1], 
                 label=lbl+"$Train$")
        ax1.plot(epochs_test, data2['ce_test'], '--', color=clrs[1], 
                 label=lbl+"$Test$")
        ax2.plot(epochs_train, data2['acc_train'], '-', color=clrs[1])
        ax2.plot(epochs_test, data2['acc_test'], '--', color=clrs[1])
        min_epochs = n_epochs if min_epochs>n_epochs else min_epochs

        # Plot data3
        lbl = "Initialized with \nuniform distribution \n range [-0.1, 0.1]\n, "
        n_epochs = len(data3['ce_train'])
        epochs_train = [ep for ep in range(n_epochs)]
        epochs_test  = [ep+1 for ep in range(n_epochs)]
        ax1.plot(epochs_train, data3['ce_train'], '-', color=clrs[2], 
                 label=lbl+"$Train$")
        ax2.plot(epochs_train, data3['acc_train'], '-', color=clrs[2])
        n_epochs = len(data3['ce_test'])
        epochs_train = [ep for ep in range(n_epochs)]
        epochs_test  = [ep+1 for ep in range(n_epochs)]
        ax1.plot(epochs_test, data3['ce_test'], '--', color=clrs[2], 
                 label=lbl+"$Test$")
        ax2.plot(epochs_test, data3['acc_test'], '--', color=clrs[2])
        min_epochs = n_epochs if min_epochs>n_epochs else min_epochs

        # Plot data4
        lbl = "Initialized with \nuniform distribution \n range [-1.0, 1.0]\n, "
        n_epochs = len(data4['ce_train'])
        epochs_train = [ep for ep in range(n_epochs)]
        epochs_test  = [ep+1 for ep in range(n_epochs)]
        ax1.plot(epochs_train, data4['ce_train'], '-', color=clrs[3], 
                 label=lbl+"$Train$")
        ax1.plot(epochs_test, data4['ce_test'], '--', color=clrs[3], 
                 label=lbl+"$Test$")
        ax2.plot(epochs_train, data4['acc_train'], '-', color=clrs[3])
        ax2.plot(epochs_test, data4['acc_test'], '--', color=clrs[3])
        min_epochs = n_epochs if min_epochs>n_epochs else min_epochs

        # Adjust axis
        ax1.legend(\
            loc='upper right', bbox_to_anchor=(1.53, 1.03),prop={'size':10})
        ax1.set_xlim([0, min_epochs])
        ax2.set_xlim([0, min_epochs])
        ax2.set_ylim([0, 1])
        ax1.set_position([0.124, 0.60, 0.775*0.73, 0.33])
        ax2.set_position([0.124, 0.10, 0.775*0.73, 0.33])

        output_file = os.path.join(\
            out_dir, "plot_lr_{}_n_hid_{}.eps".format(lr, n_hid))
        print(" Saving figure ", output_file)
        plt.savefig(output_file)

        output_file = os.path.join(\
            out_dir, "plot_lr_{}_n_hid_{}.png".format(lr, n_hid))
        print(" Saving figure ", output_file)
        plt.savefig(output_file)
        ax1.cla()
        ax2.cla()
    plt.clf()


def main():
    # Get the list of files
    input_dir1 = './initialized_with_normal'
    input_dir2 = './initialized_with_uniform_0.5'
    input_dir3 = './initialized_with_uniform_0.1'
    input_dir4 = './initialized_with_uniform_1.0'
    file_names =['n_hid_10_lr_0.1.npz', 'n_hid_10_lr_0.3.npz', 
                 'n_hid_10_lr_0.6.npz', 'n_hid_20_lr_0.1.npz', 
                 'n_hid_20_lr_0.3.npz', 'n_hid_20_lr_0.6.npz', 
                 'n_hid_30_lr_0.1.npz', 'n_hid_30_lr_0.3.npz', 
                 'n_hid_30_lr_0.6.npz']
    data_set1, data_set2, data_set3, data_set4 = [], [], [], []
    for file_name in file_names:
        file_path = os.path.join(input_dir1, file_name)
        print(" Loading ", file_path)
        data_set1.append(np.load(file_path))

        file_path = os.path.join(input_dir2, file_name)
        print(" Loading ", file_path)
        data_set2.append(np.load(file_path))

        file_path = os.path.join(input_dir3, file_name)
        print(" Loading ", file_path)
        data_set3.append(np.load(file_path))

        file_path = os.path.join(input_dir4, file_name)
        print(" Loading ", file_path)
        data_set4.append(np.load(file_path))
        
    plot_comparison(data_set1, data_set2, data_set3, data_set4, 
                    './initialization_comparison/')


if __name__=="__main__":
    main()
