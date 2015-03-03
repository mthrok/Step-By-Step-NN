import subprocess, os, sys

result_dir = "result/initialized_with_uniform_0.5"
epochs = 200
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

for n_hid in [30, 20, 10]:
    for lr in [0.6, 0.3, 0.1]:
        output = os.path.join( \
            result_dir, ("n_hid_{}_lr_{}.npz").format(n_hid, lr))
        cmd = ['python', 'back_propagation_v0.py', 
               '-hu', str(n_hid), 
               '-l', str(lr), 
               '-e', str(epochs), 
               '-i', output,
               '-o', output]
        print(*cmd)
        with subprocess.Popen(cmd) as p:
            try:
                p.wait(timeout=None)
            except KeyboardInterrupt:
                sys.exit(0)
