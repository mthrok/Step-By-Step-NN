import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import numpy as np

a = 1
b = 50

def E(x, y):
    """Ellipse-shaped error function.
    """
    x, y = x/a, y/b
    return np.square(x) + np.square(y)


def dE(x, y):
    """Compute the gradient of the error function E at x
    """
    x, y = x/a, y/b
    return 2*x, 2*y


def execute_gradient_descent(alpha=0.85, eta=0.2, x0=1.5, y0=8):
    """
    Args:
      alpha(float) : Learning rate
      eta(float) : Momentum coefficient
      x0(float), y0(float) : Initial position
      
    Returns:
      (x1, y1, e1) : The history of normal gradient descent. x and y is the 
        history of position, e is the error value.
      (x2, y2, e2) : The history of gradient descent with momentum.
      (x3, y3, e3) : The history of conjugate gradient descent.
    """
    # Gradient descent without momentum
    x1, y1, e1 = [x0], [y0], [E(x0, y0)]
    while e1[-1]>1e-10 and len(e1)<200:
        dxy = dE(x1[-1], y1[-1])
        dx = -alpha*dxy[0]
        dy = -alpha*dxy[1]
        x1.append(x1[-1]+dx)
        y1.append(y1[-1]+dy)
        e1.append(E(x1[-1], y1[-1]))
        
    # Gradient descent with momentum
    x2, y2, e2, dx, dy = [x0], [y0], [E(x0, y0)], 0, 0
    while e2[-1]>1e-10 and len(e2)<200:
        dxy = dE(x2[-1], y2[-1])
        dx = -alpha*dxy[0] + eta*dx
        dy = -alpha*dxy[1] + eta*dy
        x2.append(x2[-1]+dx)
        y2.append(y2[-1]+dy)
        e2.append(E(x2[-1], y2[-1]))    

    # Nesterov's accelarated gradient
    x3, y3, e3, dx, dy = [x0], [y0], [E(x0, y0)], 0, 0
    while e3[-1]>1e-10 and len(e3)<200:
        dxy = dE(x3[-1]+eta*dx, y3[-1]+eta*dy)
        dx = -alpha*dxy[0] + eta*dx
        dy = -alpha*dxy[1] + eta*dy
        x3.append(x3[-1]+dx)
        y3.append(y3[-1]+dy)
        e3.append(E(x3[-1], y3[-1]))

    return (x1, y1, e1), (x2, y2, e2), (x3, y3, e3)


def compare_gradient_descent(alpha, eta, plot_ax2=True,
                             ax2_xlim=(-0.01, 0.01), ax2_ylim=(-0.1, 0.1)):
    """Execute gradient descent, classic momentum and Nextrov's accelerated 
    gradient, and plot the result.
    Args:
      alpha: learning rate
      eta: momentum decay rate
    """
    (x1, y1, e1), (x2, y2, e2), (x3, y3, e3) = \
        execute_gradient_descent(alpha, eta)

    ### Plot
    # Create error surface
    x = np.arange(-2.0, 2.0, 0.1)
    y = np.arange(-9.0, 9.0, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = E(X, Y)

    fig = plt.figure()
    ax1 = fig.add_axes([0.08, 0.30, 0.38, 0.55])
    ax1.contour(X, Y, Z, levels=np.linspace(0, np.max(Z), 15), cmap=cm.gray_r)
    p2, = ax1.plot(x2, y2, 'r-o', markersize=2.0, markeredgecolor='r')
    p3, = ax1.plot(x3, y3, 'g-o', markersize=1.8, markeredgecolor='g') 
    p1, = ax1.plot(x1, y1, 'b-o', markersize=2.5, markeredgecolor='b') 
    ax1.plot([0], [0], 'kx', markersize=15)
    ax1.set_title(('Parameter movement on\nerror surface $\\frac{{\\theta_1^2}}'
                   '{{ {}^2 }}+\\frac{{\\theta_2^2}}{{ {}^2 }}$').format(a, b))
    ax1.set_xlabel('$\\theta_1$')
    ax1.set_ylabel('$\\theta_2$')

    labels=["No momentum: $\\alpha={}$".format(alpha), 
            "Classic momentum: $\\alpha={}$, $\\eta={}$".format(alpha, eta),
            ("Nesterov's accelerated gradient: $\\alpha={}$, $\\eta={}$"
             "").format(alpha, eta)]
    ax1.legend([p1, p2, p3], labels, loc='upper center', bbox_to_anchor=(1.1, -0.15))

    if plot_ax2:
        ax2 = fig.add_axes([0.28, 0.307, 0.175, 0.160])
        ax2.contour(X, Y, Z, levels=np.linspace(0, np.max(Z), 10))
        ax2.plot(x2, y2, 'r-o', markersize=2.0, markeredgecolor='r')
        ax2.plot(x3, y3, 'g-o', markersize=1.8, markeredgecolor='g')
        ax2.plot(x1, y1, 'b-o', markersize=2.5, markeredgecolor='b') 
        ax2.plot([0], [0], 'kx', markersize=15)
        ax2.set_ylim(*ax2_ylim)
        ax2.set_yticks([0.8*ax2_ylim[0], 0.0, 0.8*ax2_ylim[1]])
        plt.setp(ax2.get_yticklabels(), fontsize=8)
        ax2.set_xlim(*ax2_xlim)
        ax2.set_xticks([0.8*ax2_xlim[0], 0.0, 0.8*ax2_xlim[1]])
        ax2.xaxis.tick_top()
        # Change axis tick format to scientific
        def my_format(x, p):
            return "{:1.0e}".format(x)
        ax2.get_xaxis().set_major_formatter(ticker.FuncFormatter(my_format))
        plt.setp(ax2.get_xticklabels(), fontsize=8)

    ax3 = fig.add_subplot(122)
    ax3.plot(e3, 'g-')
    ax3.plot(e2, 'r-')
    ax3.plot(e1, 'b-')
    ax3.set_yscale('log')
    ax3.set_title('The effect of momentum on \nerror change ')
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Error")
    ax3.set_position([0.58, 0.30, 0.39, 0.55])

    print("Saving", "result/alpha_{}_eta_{}.png".format(alpha, eta))
    plt.savefig("result/alpha_{}_eta_{}.png".format(alpha, eta))


def main():
    compare_gradient_descent(0.1, 0.3, plot_ax2=False)
    compare_gradient_descent(0.1, 0.6, plot_ax2=False)
    compare_gradient_descent(0.1, 0.9, [-3e-3, 3e-3], [-0.03, 0.03])
    
    compare_gradient_descent(0.3, 0.3, plot_ax2=False)
    compare_gradient_descent(0.3, 0.6, plot_ax2=False)
    compare_gradient_descent(0.3, 0.9, [-5e-2, 5e-2], [-0.05, 0.05])
    
    compare_gradient_descent(0.6, 0.3, plot_ax2=False)
    compare_gradient_descent(0.6, 0.6, plot_ax2=False)
    compare_gradient_descent(0.6, 0.9, [-1e-2, 1e-2], [-1e-2, 1e-2])
    


if __name__=="__main__":
    main()
