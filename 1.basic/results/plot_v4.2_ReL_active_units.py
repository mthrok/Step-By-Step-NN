import matplotlib.pyplot as plt
import numpy as np
import os


def vary_colors(N):
    colors = plt.get_cmap('jet')(np.linspace(0, 1.0, N))
    if N < 7:
        for i in range(len(colors)):
            colors[i, 0:3] /= np.max(colors[i, 0:3])
    return colors


def plot_n_hids_lrs(data_sets1, output_dir):
    plt.figure()
    for data_set1 in data_sets1:
        n_hid, lr = data_set1[0]['n_hid'], data_set1[0]['lr']
        # Set up figure
        ax1 = plt.subplot(3,1,1)
        ax1.set_ylabel("Active Units [%]")
        ax1.set_title(("Unit Activity for $N_{{hid}}$:{}, $\\alpha$:{}, "
                       "NAG").format(n_hid, lr))
        ax2 = plt.subplot(3,1,2)
        ax2.set_ylabel("Cross Entropy")
        ax2.set_title(("Cross Entropy for $N_{{hid}}$:{}, $\\alpha$:{}, "
                       "NAG").format(n_hid, lr))
        ax3 = plt.subplot(3,1,3)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Accuracy")
        ax3.set_title(("Accuracy for $N_{{hid}}$:{}, $\\alpha$:{}, "
                       "NAG").format(n_hid, lr))

        # Generate colors 
        colors = vary_colors(len(data_set1))

        # Plot data
        min_epochs = np.inf
        for data, color in zip(data_set1, colors):
            lbl = "$ReL, \\ \\eta$:{}, ".format(data['mc'])
            ax1.plot(100*data['act_train'], '-.', color=color, label=lbl+"$Train$")
            ax1.plot(100*data['act_valid'], '-', color=color, label=lbl+"$Validation$")
            ax1.plot(100*data['act_test'], '--', color=color, label=lbl+"$Test$")
            ax2.plot(data['ce_valid'], '-', color=color)
            ax2.plot(data['ce_test'], '--', color=color)
            ax3.plot(data['acc_valid'], '-', color=color)
            ax3.plot(data['acc_test'], '--', color=color)
            n_epochs = len(data['ce_test'])
            min_epochs = n_epochs if min_epochs>n_epochs else min_epochs

        # Adjust axis
        ax1.legend(loc='upper right', bbox_to_anchor=(1.5, 1.03),prop={'size':10})
        ax1.set_xlim([0, min_epochs])
        ax1.set_xticklabels([])
        ax1.set_ylim([0, 105])
        ax1.yaxis.set_ticks(np.arange(0, 105, 10))
        ax1.yaxis.grid()
        ax2.set_xlim([0, min_epochs])
        ax2.set_xticklabels([])
        ax3.set_xlim([0, min_epochs])
        ax3.set_ylim([0, 1])
        ax3.yaxis.set_ticks(np.arange(0.0, 1.0, 0.1))
        ax3.yaxis.grid()

        ax1.set_position([0.124, 0.70, 0.775*0.75, 0.23])
        ax2.set_position([0.124, 0.40, 0.775*0.75, 0.23])
        ax3.set_position([0.124, 0.10, 0.775*0.75, 0.23])

        # Output figure to file
        output_file = os.path.join(\
            output_dir, "plot_v4.2_ReL_active_units_n_hid_{}_lr_{}".format(n_hid, lr))
        print(" Saving figure ", output_file+".eps")
        plt.savefig(output_file+".eps")
        print(" Saving figure ", output_file+".png")
        plt.savefig(output_file+".png")

        ax1.cla()
        ax2.cla()
        ax3.cla()
    plt.clf()


def plot(input_dir='./data', output_dir='./figure'):
    data_sets1 = []
    for n_hid in [10, 20, 30]:
        for lr in [0.1, 0.3, 0.6]:
            data1 = []
            try:
                for mc in [0.3, 0.6, 0.9]:
                    file_path = os.path.join(\
                        input_dir, 
                        "n_hid_{}_lr_{}_nag_{}_mb_ReL.npz".format(n_hid, lr, mc))
                    data1.append(np.load(file_path))
                print(" Loaded n_hid_{}, lr_{}".format(n_hid, lr))
                data_sets1.append(data1)
            except:
                pass

    plot_n_hids_lrs(data_sets1, output_dir)

if __name__=="__main__":
    plot()
