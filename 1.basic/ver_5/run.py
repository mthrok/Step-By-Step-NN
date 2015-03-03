import subprocess, os, sys

result_dir = "result"
n_epochs = 100

n_hids = [10]
lrs = [0.1, 0.3, 0.6]
mcs = [0.3, 0.6, 0.9]
wds = [0.01, 0.1, 0.0]

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

for n_hid in n_hids:
    for lr in lrs:
        for mc in mcs:
            for wd in wds:
                output = os.path.join(\
                    result_dir, ("n_hid_{}_lr_{}_nag_{}_wd_{}_mb_tanh.npz").format( \
                        n_hid, lr, mc, wd))
                cmd = ['python', 'back_propagation_v5.py', 
                       '-hu', str(n_hid), 
                       '-l', str(lr), 
                       '-m', str(mc),
                       '-w', str(wd),
                       '-e', str(n_epochs), 
                       '-i', output,
                       '-o', output]
                print(*cmd)
                with subprocess.Popen(cmd) as p:
                    try:
                        p.wait(timeout=None)
                    except KeyboardInterrupt:
                        sys.exit(0)
