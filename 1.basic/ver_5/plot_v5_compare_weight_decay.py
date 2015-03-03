from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import os, argparse


def vary_colors(N):
    colors = plt.get_cmap('jet')(np.linspace(0, 1.0, N))
    if N < 7:
        for i in range(len(colors)):
            colors[i, 0:3] /= np.max(colors[i, 0:3])
    return colors


def plot_n_hids_lrs_mcs(data_sets1, output_dir):
    plt.figure()
    for data_set1 in data_sets1:
        n_hid, lr, mc = \
            data_set1[0]['n_hid'], data_set1[0]['lr'], data_set1[0]['mc']
        # Set up figure
        ax1 = plt.subplot(3,1,1)
        ax2 = plt.subplot(3,1,2)
        ax3 = plt.subplot(3,1,3)
        ax1.set_ylabel("Cross Entropy")
        ax1.set_title(("Cross Entropy for $N_{{hid}}$:{}, $\\alpha$:{}, "
                       "$\\eta$:{}, NAG, tanh").format(n_hid, lr, mc))
        ax2.set_ylabel("Accuracy")
        ax2.set_title(("Accuracy for $N_{{hid}}$:{}, $\\alpha$:{}, "
                       "$\\eta$:{}, NAG, tanh").format(n_hid, lr, mc))
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Mean weight size")
        ax3.set_title(("Weight size for $N_{{hid}}$:{}, $\\alpha$:{}, "
                       "$\\eta$:{}, NAG, tanh").format(n_hid, lr, mc))

        # Generate colors 
        colors = vary_colors(len(data_set1))

        # Plot data
        min_epochs = np.inf
        for data, color in zip(data_set1, colors):
            lbl = "$\\lambda$:{}, ".format(data['wd'])
            ax1.plot(data['ce_valid'], '-', color=color, label=lbl+"$Validation$")
            ax1.plot(data['ce_test'], '--', color=color, label=lbl+"$Test$")
            ax2.plot(data['acc_valid'], '-', color=color)
            ax2.plot(data['acc_test'], '--', color=color)
            ax3.plot(data['size_W1']/data['W1'].size, '-', color=color, label=lbl+"W1")
            ax3.plot(data['size_W2']/data['W2'].size, '--', color=color, label=lbl+"W2")
            n_epochs = len(data['ce_test'])
            min_epochs = n_epochs if min_epochs>n_epochs else min_epochs

        # Adjust axis
        ax1.legend(loc='upper right', bbox_to_anchor=(1.355, 0.5),prop={'size':10})
        ax1.set_xlim([0, min_epochs])
        ax1.set_xticklabels([])

        ax2.set_xlim([0, min_epochs])
        ax2.set_xticklabels([])
        ax2.set_ylim([0, 1.05])
        ax2.yaxis.set_ticks(np.arange(0, 1.05, 0.1))
        ax2.yaxis.grid()

        ax3.legend(loc='upper right', bbox_to_anchor=(1.275, 1.03),prop={'size':10})
        ax3.set_xlim([0, min_epochs])
        ax3.set_yscale('log')

        ax1.set_position([0.124, 0.68, 0.775*0.83, 0.25])
        ax2.set_position([0.124, 0.38, 0.775*0.83, 0.25])
        ax3.set_position([0.124, 0.08, 0.775*0.83, 0.25])

        # Output figure to file
        output_file_name = ("plot_v5_compare_weight_decay_n_hid_{}_lr_{}_mc_{}"
                            "").format(n_hid, lr, mc)
        output_file = os.path.join(output_dir, output_file_name)
        print(" Saving figure ", output_file+".eps")
        plt.savefig(output_file+".eps")
        print(" Saving figure ", output_file+".png")
        plt.savefig(output_file+".png")

        ax1.cla()
        ax2.cla()
        ax3.cla()
    plt.clf()

def plot_weights(data_sets1, output_dir):
    # Load MNIST data
    print("Loading image/label data \n")
    mnist = datasets.fetch_mldata('MNIST Original')
    X = mnist['data']
    T = mnist['target']
    # Prepare figure and axis
    fig = plt.figure()
    for data_set1 in data_sets1:
        for data1 in data_set1:
            # Check parameters
            n_hid, lr, mc, wd = \
                data1['n_hid'], data1['lr'], data1['mc'], data1['wd']
            # Adjust paper size
            if n_hid>10:
                fig_size = [8.0,  6.0]
                fig.set_size_inches(fig_size[0], fig_size[1]*0.8*n_hid/10)

            # Show examples
            i_ax = 2
            for img, lbl in zip(X[3000:63000:6000], T[3000:63000:6000]):
                ax = fig.add_subplot(1+n_hid, 11, i_ax)
                i_ax += 1
                ax.imshow(img.reshape(28, 28), cmap=plt.cm.gray_r)
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])
                ax.set_title(str(np.asscalar(lbl)))
            # Compute filter score and
            for i_flt in range(n_hid):
                # Show filter
                ax = fig.add_subplot(1+n_hid, 11, (1+i_flt)*11+1)
                flt = data1['W1'][i_flt]
                ax.imshow(flt.reshape(28, 28), cmap=plt.cm.RdBu_r)
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])
                # Compute filter response
                print(("Computing filter responce {}/{}\r"
                       "").format(i_flt+1, n_hid), end="")
                score = np.zeros((10))
                for img, lbl in zip(X, T):
                    s = np.sum(img*flt)
                    score[int(lbl)] += s
                score /= len(X)
                ax = plt.subplot2grid((1+n_hid, 11), (1+i_flt, 1), colspan=10)
                ax.plot(score, '-')
                ax.set_xlim([-0.5, 9.5])
                ax.xaxis.set_ticks(np.linspace(0, 9, 10))
                ax.yaxis.set_ticks([])
                ax.set_yticklabels([])

            # Output figure to file
            output_file_name = ("plot_v5_view_weight_n_hid_{}_lr_{}_mc_{}_wd_{}"
                                "").format(n_hid, lr, mc, wd)
            output_file = os.path.join(output_dir, output_file_name)
            print(" Saving figure ", output_file+".eps")
            plt.savefig(output_file+".eps")
            print(" Saving figure ", output_file+".png")
            plt.savefig(output_file+".png")

    plt.clf()
            

def plot(input_dir, output_dir):
    data_sets1 = []
    for n_hid in [10, 30]:
        for lr in [0.1, 0.3, 0.6]:
            for mc in [0.3, 0.6, 0.9]:
                data1 = []
                try:
                    for wd in [0.0, 0.01, 0.1]:
                        file_name = ("n_hid_{}_lr_{}_nag_{}_wd_{}_mb_tanh"
                                     ".npz").format(n_hid, lr, mc, wd)
                        file_path = os.path.join(input_dir, file_name)
                        data1.append(np.load(file_path))
                    print(" Loaded n_hid_{}, lr_{}".format(n_hid, lr))
                    data_sets1.append(data1)
                except:
                    pass

    #plot_n_hids_lrs_mcs(data_sets1, output_dir)
    plot_weights(data_sets1, output_dir)

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input_dir', type=str, default='./data')
    p.add_argument('-o', '--output_dir', type=str, default='./figure')
    args = p.parse_args()
    plot(input_dir=args.input_dir, output_dir=args.output_dir)
