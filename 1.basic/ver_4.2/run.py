import subprocess, os, sys, argparse


def run(n_hids, lrs, mcs, n_epochs, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for n_hid in n_hids:
        for lr in lrs:
            for mc in mcs:
                output = os.path.join(\
                    output_dir, 
                    ("n_hid_{}_lr_{}_nag_{}_mb_ReL.npz").format(n_hid, lr, mc))
                cmd = ['python', 'back_propagation_v4.2.py', 
                       '-hu', str(n_hid), 
                       '-l', str(lr), 
                       '-m', str(mc), 
                       '-e', str(n_epochs), 
                       '-i', output,
                       '-o', output]
                print(*cmd)
                with subprocess.Popen(cmd) as p:
                    try:
                        p.wait(timeout=None)
                    except KeyboardInterrupt:
                        sys.exit(0)

if __name__=="__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(\
        description='Run backpropagation for the given hyperparameters.')
    parser.add_argument('-hu', '--hidden_units', type=int, nargs='+', 
                        default=[10, 20, 30], help='The numbers of hidden units')
    parser.add_argument('-l', '--learning_rates', type=float, nargs='+', 
                        default=[0.1, 0.3, 0.6], help='Learning rates')
    parser.add_argument('-e', '--epochs', type=int, 
                        default=50, help='The number of maximum epoch')
    parser.add_argument('-m', '--momentum_coefficients', type=float, nargs='+',
                        default=[0.3, 0.6, 0.9], help='Momentum coefficients')
    #parser.add_argument('-b', '--batches', type=int, default=10,
    #                    help='The number of mini-batches')
    parser.add_argument('-o', '--output-dir', type=str, 
                        default="./result")
    parser.add_argument('-p', '--parameter-sets', type=float, nargs=3, action='append',
                        help='The set of parameters (n_hid, lr, mc)')
    args = parser.parse_args()

    if args.parameter_sets:
        for p_set in args.parameter_sets:
            run(n_hids = [int(p_set[0])], 
                lrs = [p_set[1]], 
                mcs = [p_set[2]], 
                n_epochs = args.epochs, 
                output_dir = args.output_dir)
    else:
        run(n_hids = args.hidden_units, 
            lrs = args.learning_rates, 
            mcs = args.momentum_coefficients, 
            n_epochs = args.epochs, 
            output_dir = args.output_dir)
