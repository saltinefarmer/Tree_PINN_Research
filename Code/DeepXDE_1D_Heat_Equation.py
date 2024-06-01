import deepxde as dde
import numpy as np
import matplotlib.animation as ani
import matplotlib.pyplot as plt


# constants
LENGTH = 5
DURATION = 5
ALPHA = 1
BASE_TEMP = 0


# domains
time_domain = dde.geometry.TimeDomain(0, DURATION)
space_domain = dde.geometry.Interval(0, LENGTH)

domain = dde.geometry.GeometryXTime(space_domain, time_domain)


# partial differential equation
# x is 0, t is 1
def diff_EQ(inputs, u):
    u_t = dde.grad.jacobian(u, inputs, i=0, j=1) # j = 1 specifies the column where t resides
    u_xx = dde.grad.hessian(u, inputs, i=0, j=0) # j = 0 specifies the column where x resides
    return u_t - (ALPHA * u_xx)



def is_boundary(inp, on_boundary):
    return on_boundary

def is_initial(inp, on_initial):
    return on_initial

bc = dde.icbc.DirichletBC(domain, lambda t: BASE_TEMP, is_boundary)
ic = dde.icbc.IC(domain, lambda inp: 6 * np.sin((np.pi * inp[:, 0:1]) / LENGTH), is_initial)
ic2 = dde.icbc.IC(domain, lambda inp: np.sin((np.pi * inp[:, 0:1]) / LENGTH), is_initial)

conditions = [bc, ic]


train_sample = 200
bound_sample = 100
initial_sample = 100
test_sample = 500

def sol_function(inputs):
    return 6 * np.sin((np.pi * inputs[:, 0:1]) / LENGTH) * np.exp(-ALPHA * (np.square(np.pi/LENGTH)) * inputs[:, 1:2])

def sol_function2(inputs):
    return np.exp(-(np.pi**2 * ALPHA * inputs[:, 1:2]) / LENGTH**2) * np.sin((np.pi * inputs[:,0:1]) / LENGTH)


data = dde.data.TimePDE(domain, diff_EQ, conditions, num_domain=train_sample, num_boundary=bound_sample, num_initial=initial_sample, num_test=test_sample)


layer_size = [2] + [16] * 3 + [1]

activation = "tanh"
initializer = "Glorot uniform"

network = dde.nn.FNN(layer_size, activation, initializer)


model = dde.Model(data, network)

model.compile("adam", lr=.001)



losshistory, train_state = model.train(iterations=2000, display_every=500)

model.compile("L-BFGS")
losshistory, train_state = model.train()


dde.saveplot(losshistory, train_state, issave=True, isplot=True)



t_set = np.arange(0, DURATION, .1, dtype='float64')
x_set = np.arange(0, LENGTH, .1, dtype='float64')

xt_set = []

for item in t_set:
    if item != 0: 
        for item2 in x_set:
            if item2 != 0: 
                xt_set.append(np.array([item2, item]))

xt_set = np.array(xt_set)

pred = model.predict(xt_set)
true = sol_function(xt_set)


t = [xt_set[i][1] for i in range(len(xt_set))]
x = [xt_set[i][0] for i in range(len(xt_set))]



fig, ax = plt.subplots()

ax.set(xlim=[0, LENGTH], ylim=[0, 8], xlabel="Metal Rod", ylabel="Temperature")

line = ax.plot(x_set, pred[0:len(x_set)], c="k", linestyle="--", zorder=2, label="Network Prediction")[0]
line2 = ax.plot(x_set, true[0:len(x_set)], c="y", linewidth=5.0, zorder=0, label="True Value")[0]
ax.legend()


frames = []
[frames.append(item) for item in t_set if item not in frames]


def update(frame):
    temp = [pred[i] for i in range(len(t)) if t[i] == frame]
    real = [true[i] for i in range(len(t)) if t[i] == frame]

    xs = [x[i] for i in range(len(t)) if t[i] == frame]

    line.set_xdata(xs)
    line2.set_xdata(xs)
    line.set_ydata(temp)
    line2.set_ydata(real)

    return (line, line2)


animation = ani.FuncAnimation(fig=fig, func=update, frames=frames, interval=300)
plt.show()
