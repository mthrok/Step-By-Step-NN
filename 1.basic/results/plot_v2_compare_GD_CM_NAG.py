import sys, os
import numpy as np
import matplotlib.pyplot as plt


def vary_colors(N):
    colors = plt.get_cmap('jet')(np.linspace(0, 1.0, N))
    if N < 7:
        for i in range(len(colors)):
            colors[i, 0:3] /= np.max(colors[i, 0:3])
    return colors


def plot_n_hids_lrs(data_sets1, data_sets2, data_sets3, outputdir):
    plt.figure()
    # Plot each data 
    for data_set1, data_set2, data_set3 in zip(data_sets1, data_sets2, data_sets3):
        n_hid, lr = data_set1[0]['n_hid'], data_set1[0]['lr']
        # Set up figure
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross Entropy")
        ax1.set_title(\
            "Cross Entropy for $N_{{hid}}$:{}, $\\alpha$:{}".format(n_hid, lr))
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title(\
            "Accuracy for $N_{{hid}}$:{}, $\\alpha$:{}".format(n_hid, lr))
        min_epochs = np.inf

        # Generate colors
        colors = vary_colors(1+len(data_set2)+len(data_set3))

        # Plot data1
        data = data_set1[0]
        lbl = "No Momentum, "
        n_epochs = len(data['ce_train'])
        epochs = [ep for ep in range(0, n_epochs)]
        ax1.plot(epochs, data['ce_train'], '-', color=colors[0], label=lbl+"$Validation$")
        ax2.plot(epochs, data['acc_train'], '-', color=colors[0])
        min_epochs = n_epochs if min_epochs>n_epochs else min_epochs

        n_epochs = len(data['ce_test'])
        epochs  = [ep for ep in range(1, n_epochs+1)]
        ax1.plot(epochs, data['ce_test'], '--', color=colors[0], label=lbl+"$Test$")
        ax2.plot(epochs, data['acc_test'], '--', color=colors[0])
        min_epochs = n_epochs if min_epochs>n_epochs else min_epochs

        # Plot data2
        for data, color in zip(data_set2, colors[1::2]):
            lbl = "CM $\\eta$:{}, ".format(data['mc'])
            n_epochs = len(data['ce_train'])
            epochs = [ep for ep in range(0, n_epochs)]
            ax1.plot(epochs, data['ce_train'], '-', color=color, label=lbl+"$Validation$")
            ax2.plot(epochs, data['acc_train'], '-', color=color)
            min_epochs = n_epochs if min_epochs>n_epochs else min_epochs

            n_epochs = len(data['ce_test'])
            epochs  = [ep for ep in range(1, n_epochs+1)]
            ax1.plot(epochs, data['ce_test'], '--', color=color, label=lbl+"$Test$")
            ax2.plot(epochs, data['acc_test'], '--', color=color)
            min_epochs = n_epochs if min_epochs>n_epochs else min_epochs

        # Plot data3
        for data, color in zip(data_set3, colors[2::2]):
            lbl = "NAG $\\eta$:{}, ".format(data['mc'])
            n_epochs = len(data['ce_train'])
            epochs = [ep for ep in range(0, n_epochs)]
            ax1.plot(epochs, data['ce_train'], '-', color=color, label=lbl+"$Train$")
            ax2.plot(epochs, data['acc_train'], '-', color=color)
            min_epochs = n_epochs if min_epochs>n_epochs else min_epochs

            n_epochs = len(data['ce_test'])
            epochs  = [ep for ep in range(1, n_epochs+1)]
            ax1.plot(epochs, data['ce_test'], '--', color=color, label=lbl+"$Test$")
            ax2.plot(epochs, data['acc_test'], '--', color=color)
            min_epochs = n_epochs if min_epochs>n_epochs else min_epochs

        # Adjust axis
        ax1.legend(loc='upper right', bbox_to_anchor=(1.58, 1.03),prop={'size':10})
        ax1.set_xlim([0, min_epochs])
        ax2.set_xlim([0, min_epochs])
        ax2.set_ylim([0, 1])
        ax2.yaxis.set_ticks(np.arange(0.0, 1.0, 0.1))
        ax2.yaxis.grid()
        ax1.set_position([0.124, 0.60, 0.775*0.72, 0.33])
        ax2.set_position([0.124, 0.10, 0.775*0.72, 0.33])

        # Output figure to file
        output_file = os.path.join(\
            outputdir, 
            "plot_v2_compare_GD_CM_NAG_n_hid_{}_lr_{}".format(n_hid, lr))
        print(" Saving figure ", output_file+".eps")
        plt.savefig(output_file+".eps")
        print(" Saving figure ", output_file+".png")
        plt.savefig(output_file+".png")

        ax1.cla()
        ax2.cla()
    plt.clf()


def plot(input_dir='./data', output_dir='./figure'):
    data_set1, data_set2, data_set3 = [], [], []
    for n_hid in [10, 20, 30]:
        for lr in [0.1, 0.3, 0.6]:
            data1, data2, data3 = [], [], []
            try:
                file_path = os.path.join(\
                    input_dir, "n_hid_{}_lr_{}.npz".format(n_hid, lr))
                data1.append(np.load(file_path))
                for mc in [0.3, 0.6, 0.9]:
                    file_path = os.path.join(\
                        input_dir, "n_hid_{}_lr_{}_mc_{}.npz".format(n_hid, lr, mc))
                    data2.append(np.load(file_path))
                    file_path = os.path.join(\
                        input_dir, "n_hid_{}_lr_{}_nag_{}.npz".format(n_hid, lr, mc))
                    data3.append(np.load(file_path))
                print(" Loaded n_hid_{}, lr_{}".format(n_hid, lr))
                data_set1.append(data1)
                data_set2.append(data2)
                data_set3.append(data3)
            except:
                pass

    plot_n_hids_lrs(data_set1, data_set2, data_set3, output_dir)


if __name__=="__main__":
    plot()
