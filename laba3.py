import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.animation import FuncAnimation
import PyQt5
from scipy.integrate import odeint

m = 1
J = 0.5
R = 0.5
c = 2
g = 9.81
y0 = [0.5, 10, 0, 0]
t = np.linspace(0, 1000, 10000)


def SystemOfEquations(y, t):
    global m, J, R, c, g
    yt = np.zeros_like(y)

    yt[0] = y[2]
    yt[1] = y[3]

    a11 = J + m * R ** 2 * np.sin(y[2]) ** 2
    a12 = 0
    b1 = -c * y[0] - m * R ** 2 * y[2] * y[3] * np.sin(2 * y[1])

    a21 = 0
    a22 = R
    b2 = -g * np.sin(y[1]) + R * y[2] ** 2 * np.sin(y[1]) * np.cos(y[1])

    A = [[a11, a12], [a21, a22]]
    A1 = [[b1, a12], [b2, a22]]
    A2 = [[b1, a12], [b2, a22]]

    yt[2] = np.linalg.det(A1) / np.linalg.det(A)
    yt[3] = np.linalg.det(A2) / np.linalg.det(A)

    return yt


def Animation(t):
    global Phi, Tetta

    trubka.set_data_3d(R * np.sin(angle) * np.cos(Phi[t]),
                       R * np.sin(angle) * np.sin(Phi[t]), z)

    Point.set_data_3d(R * np.sin(Tetta[t]) * np.cos(Phi[t]),
                      R * np.sin(Tetta[t]) * np.sin(Phi[t]),
                      R * np.cos(Tetta[t]))

    AB.set_data_3d([0, R * np.sin(Tetta[t]) * np.cos(Phi[t])],
                   [0, R * np.sin(Tetta[t]) * np.sin(Phi[t])],
                   [0, R * np.cos(Tetta[t])])

    return [trubka, Point, AB]


def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

   Make axes of 3D plot have equal scale so that spheres appear as
   spheres and cubes as cubes.  Required since `ax.axis('equal')`
   and `ax.set_aspect('equal')` don't work on 3D.
   """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


if __name__ == "__main__":
    matplotlib.use("Qt5Agg")
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    res = odeint(SystemOfEquations, y0, t)

    Phi = res[:, 0]
    Tetta = res[:, 1]
    Phit = res[:, 2]
    Tettat = res[:, 3]

    angle = np.linspace(0, 2 * np.pi, 150)
    radius = R

    x = R * np.sin(angle) * np.cos(Phi[0])
    y = -R * np.sin(angle) * np.sin(Phi[0])
    z = -R * np.cos(angle)

    trubka = ax.plot(x, y, z)[0]

    Vertical = ax.plot([0, 0], [0, 0], [-0.9, 0.9], color=[0, 0, 0], linestyle='dashed')[0]

    Point = ax.plot(x[0], y[0], z[0], color=[1, 0, 0], marker='o', markersize=10)[0]

    Point2 = ax.plot(0, 0, 0, color=[0, 1, 0], marker='o')[0]
    AB = ax.plot([0, x[0]], [0, y[0]], [0, z[0]], color=[0, 0, 0], linestyle='dotted')[0]

    a = FuncAnimation(fig=fig, func=Animation, frames=len(x), blit=True, interval=100)

    A = ax.plot(0, 0, 0.5, marker='o', color="green")
    ax.text(0.1, 0.1, 0.6, "A")

    D = ax.plot(0, 0, -0.5, marker='o', color="green")
    ax.text(0.1, 0.1, -0.6, "D")

    ax.text(0.1, 0.1, 0.1, "O1")

    O = ax.plot(0, 0, -0.9, marker='o', color="brown")
    ax.text(0.1, 0.1, -1, "O")

    B = ax.plot(0, 0, 0.9, marker='o', color="brown")
    ax.text(0.1, 0.1, 1, "B")

    plt.show()