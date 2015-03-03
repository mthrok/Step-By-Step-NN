import subprocess, os, sys

result_dir = "result"
n_epochs = 200

n_hids = [10, 20, 30]
lrs = [0.1, 0.3, 0.6]
mcs = [0.3, 0.6, 0.9]

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# TODO
# Make the initial weight same 

for n_hid in n_hids:
    for lr in lrs:
        for mc in mcs:
            output = os.path.join(\
                result_dir, ("n_hid_{}_lr_{}_nag_{}.npz").format( \
                    n_hid, lr, mc))
            cmd = ['python', 'back_propagation_v2.py', 
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
