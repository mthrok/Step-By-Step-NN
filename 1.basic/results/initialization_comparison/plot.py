import sys, os
import numpy as np
import matplotlib.pyplot as plt


def vary_colors(N):
    colors = plt.get_cmap('jet')(np.linspace(0, 1.0, N))
    if N<7:
        for i in range(len(colors)):
            colors[i, 0:3] /= np.max(colors[i, 0:3])
    return colors


def plot_n_hids(data_set, outputdir):
    plt.figure()
    # Get the list of n_hids
    n_hids = set([])
    for data in data_set:
        n_hids.add(np.asscalar(data['n_hid']))

    # Plot each data 
    for n_hid in n_hids:
        # Set up figure
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross Entropy")
        ax1.set_title("Cross Entropy for $N_{{hid}}$:{}".format(n_hid))
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy for $N_{{hid}}$:{}".format(n_hid))
        min_epochs = np.inf

        # Check the number of lines
        N = 0
        for data in data_set:
            if n_hid==np.asscalar(data['n_hid']):
                N += 1

        # Generate colors
        clrs, c = vary_colors(N), 0
        # Plot data
        for data in data_set:
            if not n_hid==np.asscalar(data['n_hid']):
                continue

            lbl = "$\\alpha$:{}, ".format(data['lr'])
            n_epochs = len(data['ce_train'])
            epochs_train = [ep for ep in range(0, n_epochs)]
            epochs_test  = [ep for ep in range(1, n_epochs+1)]
            ax1.plot(epochs_train, data['ce_train'], '-', color=clrs[c], 
                     label=lbl+"$Train$")
            ax2.plot(epochs_train, data['acc_train'], '-', color=clrs[c])

            n_epochs = len(data['ce_test'])
            epochs_train = [ep for ep in range(0, n_epochs)]
            epochs_test  = [ep for ep in range(1, n_epochs+1)]
            ax1.plot(epochs_test, data['ce_test'], '--', color=clrs[c], 
                     label=lbl+"$Test$")
            ax2.plot(epochs_test, data['acc_test'], '--', color=clrs[c])
            c += 1
            min_epochs = n_epochs if min_epochs>n_epochs else min_epochs
        # Adjust axis
        ax1.legend(loc='upper right', 
                   bbox_to_anchor=(1.4, 1.03),prop={'size':12})
        ax1.set_xlim([0, min_epochs])
        ax2.set_xlim([0, min_epochs])
        ax2.set_ylim([0, 1])
        ax1.set_position([0.124, 0.60, 0.775*0.80, 0.33])
        ax2.set_position([0.124, 0.10, 0.775*0.80, 0.33])

        # Output figure to file
        output_file = os.path.join(outputdir, "plot_n_hid_{}.eps".format(n_hid))
        print(" Saving figure ", output_file)
        plt.savefig(output_file)
        output_file = os.path.join(outputdir, "plot_n_hid_{}.png".format(n_hid))
        print(" Saving figure ", output_file)
        plt.savefig(output_file)
        ax1.cla()
        ax2.cla()
    plt.clf()


def plot_lrs(data_set, outputdir):
    plt.figure()
    # Get the list of lrs
    lrs = set([])
    for data in data_set:
        lrs.add(np.asscalar(data['lr']))

    # Plot each data 
    for lr in lrs:
        # Set up figure
        ax1 = plt.subplot(2,1,1)
        ax2 = plt.subplot(2,1,2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross Entropy")
        ax1.set_title("Cross Entropy for Learning Rate $\\alpha$:{}".format(lr))
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy for Learning Rate $\\alpha$:{}".format(lr))
        min_epochs = np.inf

        # Check the number of lines
        N = 0
        for data in data_set:
            if lr==np.asscalar(data['lr']):
                N += 1

        # Generate colors
        clrs, c = vary_colors(N), 0
        # Plot data
        for data in data_set:
            if not lr==np.asscalar(data['lr']):
                continue

            lbl = "$N_{{hid}}:{}$, ".format(data['n_hid'])
            ax1.plot(data['ce_train'], '-', color=clrs[c], label=lbl+"$Train$")
            ax1.plot(data['ce_test'], '--', color=clrs[c], label=lbl+"$Test$")
            ax2.plot(data['acc_train'], '-', color=clrs[c])
            ax2.plot(data['acc_test'], '--', color=clrs[c])
            c += 1
            min_epochs = min_epochs \
                if min_epochs<len(data['ce_test']) else len(data['ce_test'])
        # Adjust axis
        ax1.legend(loc='upper right', bbox_to_anchor=(1.4, 1.03),prop={'size':12})
        ax1.set_xlim([0, min_epochs])
        ax2.set_xlim([0, min_epochs])
        ax2.set_ylim([0, 1])
        ax1.set_position([0.124, 0.60, 0.775*0.80, 0.33])
        ax2.set_position([0.124, 0.10, 0.775*0.80, 0.33])

        output_file = os.path.join(outputdir, "plot_lr_{}.eps".format(lr))
        print(" Saving figure ", output_file)
        plt.savefig(output_file)
        output_file = os.path.join(outputdir, "plot_lr_{}.png".format(lr))
        print(" Saving figure ", output_file)
        plt.savefig(output_file)
        ax1.cla()
        ax2.cla()
    plt.clf()


def plot(input_dir):
    data_set = []
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".npz"):
            try:
                file_path = os.path.join(input_dir, file_name)
                print(" Loading ", file_path)
                data = np.load(file_path)
                data_set.append(data)
            except:
                print(" Failed to load ", file)

    plot_lrs(data_set, input_dir)
    plot_n_hids(data_set, input_dir)

if __name__=="__main__":
    plot('initialized_with_normal')
    plot('initialized_with_uniform_0.1')
    plot('initialized_with_uniform_0.5')
    plot('initialized_with_uniform_1.0')
