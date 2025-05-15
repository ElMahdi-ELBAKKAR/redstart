import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Redstart: A Lightweight Reusable Booster""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="public/images/redstart.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Project Redstart is an attempt to design the control systems of a reusable booster during landing.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In principle, it is similar to SpaceX's Falcon Heavy Booster.

    >The Falcon Heavy booster is the first stage of SpaceX's powerful Falcon Heavy rocket, which consists of three modified Falcon 9 boosters strapped together. These boosters provide the massive thrust needed to lift heavy payloads—like satellites or spacecraft—into orbit. After launch, the two side boosters separate and land back on Earth for reuse, while the center booster either lands on a droneship or is discarded in high-energy missions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(
        mo.Html("""
    <iframe width="560" height="315" src="https://www.youtube.com/embed/RYUr-5PYA7s?si=EXPnjNVnqmJSsIjc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>""")
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell(hide_code=True)
def _():
    import scipy
    import scipy.integrate as sci

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    from tqdm import tqdm

    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple
    return FFMpegWriter, FuncAnimation, mpl, np, plt, scipy, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The Model

    The Redstart booster in model as a rigid tube of length $2 \ell$ and negligible diameter whose mass $M$ is uniformly spread along its length. It may be located in 2D space by the coordinates $(x, y)$ of its center of mass and the angle $\theta$ it makes with respect to the vertical (with the convention that $\theta > 0$ for a left tilt, i.e. the angle is measured counterclockwise)

    This booster has an orientable reactor at its base ; the force that it generates is of amplitude $f>0$ and the angle of the force with respect to the booster axis is $\phi$ (with a counterclockwise convention).

    We assume that the booster is subject to gravity, the reactor force and that the friction of the air is negligible.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image(src="public/images/geometry.svg"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Constants

    For the sake of simplicity (this is merely a toy model!) in the sequel we assume that: 

      - the total length $2 \ell$ of the booster is 2 meters,
      - its mass $M$ is 1 kg,
      - the gravity constant $g$ is 1 m/s^2.

    This set of values is not realistic, but will simplify our computations and do not impact the structure of the booster dynamics.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Helpers

    ### Rotation matrix

    $$ 
    \begin{bmatrix}
    \cos \alpha & - \sin \alpha \\
    \sin \alpha &  \cos \alpha  \\
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return (R,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Videos

    It will be very handy to make small videos to visualize the evolution of our booster!
    Here is an example of how such videos can be made with Matplotlib and displayed in marimo.
    """
    )
    return


@app.cell(hide_code=True)
def _(FFMpegWriter, FuncAnimation, mo, np, plt, tqdm):
    def make_video(output):
        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        num_frames = 100
        fps = 30 # Number of frames per second

        def animate(frame_index):    
            # Clear the canvas and redraw everything at each step
            plt.clf()
            plt.xlim(0, 2*np.pi)
            plt.ylim(-1.5, 1.5)
            plt.title(f"Sine Wave Animation - Frame {frame_index+1}/{num_frames}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)

            x = np.linspace(0, 2*np.pi, 100)
            phase = frame_index / 10
            y = np.sin(x + phase)
            plt.plot(x, y, "r-", lw=2, label=f"sin(x + {phase:.1f})")
            plt.legend()

            pbar.update(1)

        pbar = tqdm(total=num_frames, desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=num_frames)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")

    _filename = "wave_animation.mp4"
    make_video(_filename)
    mo.show_code(mo.video(src=_filename))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Getting Started""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Constants

    Define the Python constants `g`, `M` and `l` that correspond to the gravity constant, the mass and half-length of the booster.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    g = 1.0
    M = 1.0
    l = 1
    return M, g, l


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Forces

    Compute the force $(f_x, f_y) \in \mathbb{R}^2$ applied to the booster by the reactor.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    f_x & = -f \sin (\theta + \phi) \\
    f_y & = +f \cos(\theta +\phi)
    \end{align*}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Center of Mass

    Give the ordinary differential equation that governs $(x, y)$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    \begin{align*}
    M \ddot{x} & = -f \sin (\theta + \phi) \\
    M \ddot{y} & = +f \cos(\theta +\phi) - Mg
    \end{align*}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Moment of inertia

    Compute the moment of inertia $J$ of the booster and define the corresponding Python variable `J`.
    """
    )
    return


@app.cell
def _(M, l):
    J = M * l * l / 3
    J
    return (J,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Tilt

    Give the ordinary differential equation that governs the tilt angle $\theta$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    J \ddot{\theta} = - \ell (\sin \phi)  f
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Simulation

    Define a function `redstart_solve` that, given as input parameters: 

      - `t_span`: a pair of initial time `t_0` and final time `t_f`,
      - `y0`: the value of the state `[x, dx, y, dy, theta, dtheta]` at `t_0`,
      - `f_phi`: a function that given the current time `t` and current state value `y`
         returns the values of the inputs `f` and `phi` in an array.

    returns:

      - `sol`: a function that given a time `t` returns the value of the state `[x, dx, y, dy, theta, dtheta]` at time `t` (and that also accepts 1d-arrays of times for multiple state evaluations).

    A typical usage would be:

    ```python
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    ```

    Test this typical example with your function `redstart_solve` and check that its graphical output makes sense.
    """
    )
    return


@app.cell(hide_code=True)
def _(J, M, g, l, np, scipy):
    def redstart_solve(t_span, y0, f_phi):
        def fun(t, state):
            x, dx, y, dy, theta, dtheta = state
            f, phi = f_phi(t, state)
            d2x = (-f * np.sin(theta + phi)) / M
            d2y = (+ f * np.cos(theta + phi)) / M - g
            d2theta = (- l * np.sin(phi)) * f / J
            return np.array([dx, d2x, dy, d2y, dtheta, d2theta])
        r = scipy.integrate.solve_ivp(fun, t_span, y0, dense_output=True)
        return r.sol
    return (redstart_solve,)


@app.cell(hide_code=True)
def _(l, np, plt, redstart_solve):
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0) = - 2*\ell$,  can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    % y(t)
    y(t)
    = \frac{2(5-\ell)}{125}\,t^3
      + \frac{3\ell-10}{25}\,t^2
      - 2\,t
      + 10
    $$

    $$
    % f(t)
    f(t)
    = M\!\Bigl[
        \frac{12(5-\ell)}{125}\,t
        + \frac{6\ell-20}{25}
        + g
      \Bigr].
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(M, g, l, np, plt, redstart_solve):

    def smooth_landing_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example()
    return


@app.cell
def _(M, g, l, np, plt, redstart_solve):
    def smooth_landing_example_force():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example_force()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Drawing

    Create a function that draws the body of the booster, the flame of its reactor as well as its target landing zone on the ground (of coordinates $(0, 0)$).

    The drawing can be very simple (a rectangle for the body and another one of a different color for the flame will do perfectly!).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image("public/images/booster_drawing.png"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Make sure that the orientation of the flame is correct and that its length is proportional to the force $f$ with the length equal to $\ell$ when $f=Mg$.

    The function shall accept the parameters `x`, `y`, `theta`, `f` and `phi`.
    """
    )
    return


@app.cell(hide_code=True)
def _(M, R, g, l, mo, mpl, np, plt):
    def draw_booster(x=0, y=l, theta=0.0, f=0.0, phi=0.0, axes=None, **options):
        L = 2 * l
        if axes is None:
            _fig, axes = plt.subplots()

        axes.set_facecolor('#F0F9FF') 

        ground = np.array([[-2*l, 0], [2*l, 0], [2*l, -l], [-2*l, -l], [-2*l, 0]]).T
        axes.fill(ground[0], ground[1], color="#E3A857", **options)

        b = np.array([
            [l/10, -l], 
            [l/10, l], 
            [0, l+l/10], 
            [-l/10, l], 
            [-l/10, -l], 
            [l/10, -l]
        ]).T
        b = R(theta) @ b
        axes.fill(b[0]+x, b[1]+y, color="black", **options)

        ratio = l / (M*g) # when f= +MG, the flame length is l 

        flame = np.array([
            [l/10, 0], 
            [l/10, - ratio * f], 
            [-l/10, - ratio * f], 
            [-l/10, 0], 
            [l/10, 0]
        ]).T
        flame = R(theta+phi) @ flame
        axes.fill(
            flame[0] + x + l * np.sin(theta), 
            flame[1] + y - l * np.cos(theta), 
            color="#FF4500", 
            **options
        )

        return axes

    _axes = draw_booster(x=0.0, y=20*l, theta=np.pi/8, f=M*g, phi=np.pi/8)
    _fig = _axes.figure
    _axes.set_xlim(-4*l, 4*l)
    _axes.set_ylim(-2*l, 24*l)
    _axes.set_aspect("equal")
    _axes.grid(True)
    _MaxNLocator = mpl.ticker.MaxNLocator
    _axes.xaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.yaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.set_axisbelow(True)
    mo.center(_fig)
    return (draw_booster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualisation

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin the with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


@app.cell(hide_code=True)
def _(draw_booster, l, mo, np, plt, redstart_solve):
    def sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_1()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_2():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_2()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_3()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_4():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_4()
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    draw_booster,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_1.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_1())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_2():
        L = 2*l

        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_2.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_2())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_3.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_3())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_4():
        L = 2*l
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_4.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_4())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Linearized Dynamics""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Equilibria

    We assume that $|\theta| < \pi/2$, $|\phi| < \pi/2$ and that $f > 0$. What are the possible equilibria of the system for constant inputs $f$ and $\phi$ and what are the corresponding values of these inputs?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ⭐ Answer

    We assume the system is at *equilibrium*, which means:

    - All derivatives are zero:
      - \( \dot{x} = \dot{y} = \dot{\theta} = 0 \)
      - \( \ddot{x} = \ddot{y} = \ddot{\theta} = 0 \)

    So we set the equations of motion to zero.


    #### 1. No lateral acceleration:

    \[
    M \ddot{x} = -f \sin(\theta + \phi) = 0
    \Rightarrow \sin(\theta + \phi) = 0
    \Rightarrow \theta + \phi = 0 \quad (\text{mod } \pi)
    \]

    Since \( |\theta| < \pi/2 \) and \( |\phi| < \pi/2 \), we must have:

    \[
    \boxed{\phi = -\theta}
    \]


    #### 2. No vertical acceleration:

    \[
    M \ddot{y} = f \cos(\theta + \phi) - Mg = 0
    \Rightarrow f = Mg \cos(\theta + \phi)
    \]

    Using \( \theta + \phi = 0 \Rightarrow \cos(\theta + \phi) = 1 \), we find:

    \[
    \boxed{f = Mg}
    \]


    #### 3. No rotational acceleration:

    \[
    J \ddot{\theta} = -\ell f \sin(\phi) = 0
    \Rightarrow \sin(\phi) = 0 \Rightarrow \phi = 0
    \]

    But we also had \( \phi = -\theta \), so:

    \[
    \boxed{\theta = \phi = 0}
    \]


    ### Conclusion:

    The only possible equilibrium under these constraints is:

    - \( \boxed{\phi = 0} \) (thrust aligned)
    - \( \boxed{f = Mg} \) (force cancels gravity)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linearized Model

    Introduce the error variables $\Delta x$, $\Delta y$, $\Delta \theta$, and $\Delta f$ and $\Delta \phi$ of the state and input values with respect to the generic equilibrium configuration.
    What are the linear ordinary differential equations that govern (approximately) these variables in a neighbourhood of the equilibrium?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ⭐ Answer

    Using the Taylor expansions of the dynamics near equilibrium, and assuming small \( \theta \) and \( \phi \):

    \[
    \begin{aligned}
    \ddot{x} &= -\frac{f}{M} \sin(\theta + \phi) \approx -g (\Delta \theta + \Delta \phi) \\
    \ddot{y} &= \frac{f}{M} \cos(\theta + \phi) - g \approx \frac{\Delta f}{M} \\
    \ddot{\theta} &= -\frac{\ell f}{J} \sin(\phi) \approx -\frac{Mg \ell}{J} \Delta \phi
    \end{aligned}
    \]

    So, the *linearized ODE system* is:

    \[
    \begin{aligned}
    \Delta \ddot{x} &= -g (\Delta \theta + \Delta \phi) \\
    \Delta \ddot{y} &= \frac{1}{M} \Delta f \\
    \Delta \ddot{\theta} &= -\frac{Mg \ell}{J} \Delta \phi
    \end{aligned}
    \]

    This gives the approximate evolution of the system in a small neighborhood of the equilibrium.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Standard Form

    What are the matrices $A$ and $B$ associated to this linear model in standard form?
    Define the corresponding NumPy arrays `A` and `B`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ⭐ Answer

    We linearized the system around the equilibrium:

    - \( \theta = 0 \), \( \phi = 0 \), \( f = Mg \)
    - Small angle approximations: \( \sin(\theta + \phi) \approx \theta + \phi \), \( \cos(\theta + \phi) \approx 1 \)

    From the linearized dynamics:

    \[
    \begin{aligned}
    \Delta \ddot{x} &= -g (\Delta \theta + \Delta \phi) \\
    \Delta \ddot{y} &= \frac{1}{M} \Delta f \\
    \Delta \ddot{\theta} &= -\frac{Mg \ell}{J} \Delta \phi
    \end{aligned}
    \]

    We define the state vector:

    \[
    \Delta X =
    \begin{bmatrix}
    \Delta x \\
    \Delta \dot{x} \\
    \Delta y \\
    \Delta \dot{y} \\
    \Delta \theta \\
    \Delta \dot{\theta}
    \end{bmatrix}, \quad
    \Delta \dot{X} =
    \begin{bmatrix}
    \Delta \dot{x} \\
    \Delta \ddot{x} \\
    \Delta \dot{y} \\
    \Delta \ddot{y} \\
    \Delta \dot{\theta} \\
    \Delta \ddot{\theta}
    \end{bmatrix}
    \]

    And the input vector:

    \[
    \Delta u =
    \begin{bmatrix}
    \Delta f \\
    \Delta \phi
    \end{bmatrix}
    \]

    So the system can be written as:

    \[
    \Delta \dot{X} = A \Delta X + B \Delta u
    \]

    ### Matrice A

    $$
    A =
    \begin{bmatrix}
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & -{g} & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
    $$

    ### Matrice B

    $$
    B =
    \begin{bmatrix}
    0 & 0 \\
    0 & -{g} \\
    0 & 0 \\
    \frac{1}{{{M}}} & 0 \\
    0 & 0 \\
    0 & {-\frac{Mgl}{J}}
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell
def _(J, M, g, l, np):
    # Precompute gain for d²θ equation
    a = M * g * l / J  # = (1 * 1 * 1) / (1 * 1² / 3) = 3

    # State matrix A
    A = np.array([
        [0, 1, 0, 0,  0, 0],
        [0, 0, 0, 0, -g, 0],
        [0, 0, 0, 1,  0, 0],
        [0, 0, 0, 0,  0, 0],
        [0, 0, 0, 0,  0, 1],
        [0, 0, 0, 0,  0, 0]
    ])

    # Input matrix B
    B = np.array([
        [0,     0],
        [0,   -g],
        [0,     0],
        [1/M,   0],
        [0,     0],
        [0,   -a]
    ])

    A, B
    return A, B, a


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Stability

    Is the generic equilibrium asymptotically stable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ⭐ Answer

    To determine whether the generic equilibrium is asymptotically stable, we analyze the eigenvalues of the linearized system matrix \( A \) around that equilibrium.

    We find:

    \[
    \text{Eig}(A) = \{0, 0, 0, 0, 0, 0\}
    \]

    This means that all eigenvalues are exactly zero. Therefore the system is not asymptotically stable.
    """
    )
    return


@app.cell
def _(A, np):


    eigvals = np.linalg.eigvals(A)
    print("Eigenvalues:", eigvals)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controllability

    Is the linearized model controllable?
    """
    )
    return


app._unparsable_cell(
    r"""
    ## ⭐ Answer
    ## Controllability of the System

    The system is controllable if the controllability matrix:

    $$
    \mathcal{C} = \begin{bmatrix}
    B & AB & A^2 B & \cdots & A^{n-1} B
    \end{bmatrix}
    $$

    has full rank, i.e 

    $$
    \text{rank}(\mathcal{C}) = n
    $$

    where \( n = 6 \) is the number of state variables.
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


@app.cell
def _(A, B, np):

    from numpy.linalg import matrix_rank


    n = A.shape[0]
    controllability_matrix = B
    for i in range(1, n):
        controllability_matrix = np.hstack((controllability_matrix, np.linalg.matrix_power(A, i) @ B))
    rank = matrix_rank(controllability_matrix)
    print(f"Rank of controllability matrix: {rank}")
    print("System is controllable" if rank == n else "System is NOT controllable")
    return (matrix_rank,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Lateral Dynamics

    We limit our interest in the lateral position $x$, the tilt $\theta$ and their derivatives (we are for the moment fine with letting $y$ and $\dot{y}$ be uncontrolled). We also set $f = M g$ and control the system only with $\phi$.

    What are the new (reduced) matrices $A$ and $B$ for this reduced system?
    Check the controllability of this new system.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ⭐ Answer

    We reduce our analysis to the lateral dynamics:

    - *State variables*: \( \Delta x, \Delta \dot{x}, \Delta \theta, \Delta \dot{\theta} \)

    - *Input*: \( \Delta \phi \)

    - Assume \( f = Mg \) is constant

    ---

    ### Linearized Equations

    From the full linearized model:

    \[
    \begin{aligned}
    \Delta \ddot{x} &= -g (\Delta \theta + \Delta \phi) \\
    \Delta \ddot{\theta} &= -\frac{Mg\ell}{J} \Delta \phi
    \end{aligned}
    \]

    We rewrite this in *state-space form* with:

    \[
    \Delta X_r = 
    \begin{bmatrix}
    \Delta x \\
    \Delta \dot{x} \\
    \Delta \theta \\
    \Delta \dot{\theta}
    \end{bmatrix}, \quad
    \Delta u_r = \Delta \phi
    \]

    The state-space model is:

    \[
    \dot{\Delta X_r} = A_r \Delta X_r + B_r \Delta u_r
    \]

    With:

    \[
    A_r =
    \begin{bmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0
    \end{bmatrix}, \quad
    B_r =
    \begin{bmatrix}
    0 \\
    -g \\
    0 \\
    -\dfrac{Mg\ell}{J}
    \end{bmatrix}
    \]

    Using \( M = 1 \), \( g = 1 \), \( \ell = 1 \), \( J = \frac{1}{3} \), we get:

    \[
    B_r =
    \begin{bmatrix}
    0 \\
    -1 \\
    0 \\
    -3
    \end{bmatrix}
    \]

    Conclusion

    The *reduced state-space model* is:

    \[
    \dot{\Delta X_r} = A_r \Delta X_r + B_r \Delta \phi
    \]

    With:

    \[
    A_r =
    \begin{bmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & -1 & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0
    \end{bmatrix}, \quad
    B_r =
    \begin{bmatrix}
    0 \\
    -1 \\
    0 \\
    -3
    \end{bmatrix}
    \]

    We computed the *rank of the controllability matrix*:

    \[
    \boxed{\text{rank} = 4}
    \]

    ---

    ### Final Answer

     *Yes, the reduced system is **fully controllable* with the input \( \Delta \phi \).
    """
    )
    return


@app.cell
def _(a, g, matrix_rank, np):


    # Reduced A and B matrices
    Ar = np.array([
        [0, 1, 0, 0],
        [0, 0, -g, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    Br = np.array([
        [0],
        [-g],
        [0],
        [-a]
    ])

    # Controllability matrix
    Cr = np.hstack([Br, Ar @ Br, Ar @ Ar @ Br, Ar @ Ar @ Ar @ Br])
    rank_Cr = matrix_rank(Cr)

    # Output
    print("Reduced A matrix:\n", Ar)
    print("\nReduced B matrix:\n", Br)
    print("\nControllability matrix rank:", rank_Cr)




    # Reduced A and B matrices
    Ar = np.array([
        [0, 1, 0, 0],
        [0, 0, -g, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    Br = np.array([
        [0],
        [-g],
        [0],
        [-a]
    ])

    # Controllability matrix
    Cr = np.hstack([Br, Ar @ Br, Ar @ Ar @ Br, Ar @ Ar @ Ar @ Br])
    rank_Cr = matrix_rank(Cr)

    # Output
    print("Reduced A matrix:\n", Ar)
    print("\nReduced B matrix:\n", Br)
    print("\nControllability matrix rank:", rank_Cr)
    return Ar, Br


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linear Model in Free Fall

    Make graphs of $y(t)$ and $\theta(t)$ for the linearized model when $\phi(t)=0$,
    $x(0)=0$, $\dot{x}(0)=0$, $\theta(0) = 45 / 180  \times \pi$  and $\dot{\theta}(0) =0$. What do you see? How do you explain it?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## ⭐ Answer

    ### Observations from the Plot

    1. **Uncontrolled Descent:**
       - The \(y(t)\) (blue curve) decreases rapidly over time, indicating a free fall.
       - This is expected since without a compensating vertical force, gravity dominates the vertical motion.

    2. **Tilt Behavior:**
       - The \( \theta(t) \) (red curve) remains constant due to the null nature of its derivatives.
    """
    )
    return


@app.cell
def _(g, np, plt):
    from scipy.integrate import solve_ivp

    # Initial conditions
    theta0 = np.pi / 4   
    theta_dot0 = 0       
    y0 = 0               
    y_dot0 = 0           

    def linear_model(t, state):
        y, y_dot, theta, theta_dot = state
        return [
            y_dot,         # y'
            -g,            # y'' = -g
            theta_dot,     # θ'
            0              # θ'' = 0, since ϕ = 0
        ]

    initial_state = [y0, y_dot0, theta0, theta_dot0]

    t_vals = np.linspace(0, 5, 500)

    sol = solve_ivp(linear_model, [0, 5], initial_state, t_eval=t_vals)

    y_vals = sol.y[0]
    theta_vals = sol.y[2]

    plt.figure(figsize=(10, 5))
    plt.plot(t_vals, y_vals, label="y(t)", color='blue')
    plt.plot(t_vals, theta_vals, label=r"$\theta(t)$", color='red')
    plt.xlabel("Time (s)")
    plt.ylabel("Values")
    plt.title("y(t) and θ(t) for the Linearized Model")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return (solve_ivp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Manually Tuned Controller

    Try to find the two missing coefficients of the matrix 

    $$
    K =
    \begin{bmatrix}
    0 & 0 & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    manages  when
    $\Delta x(0)=0$, $\Delta \dot{x}(0)=0$, $\Delta \theta(0) = 45 / 180  \times \pi$  and $\Delta \dot{\theta}(0) =0$ to: 

      - make $\Delta \theta(t) \to 0$ in approximately $20$ sec (or less),
      - $|\Delta \theta(t)| < \pi/2$ and $|\Delta \phi(t)| < \pi/2$ at all times,
      - (but we don't care about a possible drift of $\Delta x(t)$).

    Explain your thought process, show your iterations!

    Is your closed-loop model asymptotically stable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ⭐ Answer

    Nous cherchons à compléter le vecteur de gain :

    \[
    K = [0,\ 0,\ k_3,\ k_4]
    \]

    dans la loi de commande :

    \[
    \Delta \phi(t) = -k_3 \Delta \theta(t) - k_4 \Delta \dot{\theta}(t)
    \]

    avec les objectifs suivants :

    - \( \Delta \theta(t) \to 0 \) en 20 secondes ou moins
    - \( |\Delta \theta(t)| < \frac{\pi}{2} \)
    - \( |\Delta \phi(t)| < \frac{\pi}{2} \)
    - Pas de contrainte sur \( \Delta x(t) \)

    Les conditions initiales sont :

    \[
    \Delta x(0) = 0, \quad \Delta \dot{x}(0) = 0, \quad \Delta \theta(0) = \frac{\pi}{4}, \quad \Delta \dot{\theta}(0) = 0
    \]

    Nous utilisons le système réduit suivant :

    \[
    \dot{X}_r = A_r X_r + B_r \Delta \phi
    \]

    où :

    \[
    X_r = \begin{bmatrix} \Delta x \\ \Delta \dot{x} \\ \Delta \theta \\ \Delta \dot{\theta} \end{bmatrix}, \quad
    A_r = \begin{bmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & -1 & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0
    \end{bmatrix}, \quad
    B_r = \begin{bmatrix}
    0 \\
    -1 \\
    0 \\
    -3
    \end{bmatrix}
    \]

    On choisit : 

    \[
    K = [0,\ 0,\ 10,\ 6]
    \]

    et on construit :

    \[
    A_{\text{cl}} = A_r - B_r K
    \]

    ---

    Les valeurs propres du système en boucle fermée sont :

    - Une valeur propre *positive*
    - Une valeur propre *nulle*
    - Deux valeurs propres complexes à *partie réelle négative*

    Cela signifie que :

    - Le système *n’est pas asymptotiquement stable*
    - Une partie de la dynamique diverge
    - Même si \( \theta(t) \) semble converger au début, le système global *peut diverger à long terme*

    ---


    Le retour d’état choisi *ne contrôle pas les états* \( \Delta x \) et \( \Delta \dot{x} \), or :

    \[
    \Delta \ddot{x} = -g(\Delta \theta + \Delta \phi)
    \]

    donc des variations de \( \theta \) ont un impact direct sur la dynamique de \( x \), qui *reste non contrôlée, ce qui crée une **instabilité globale*.

    ### Conclusion

    Avec :

    \[
    K = [0,\ 0,\ 10,\ 6]
    \]

    le système n’est *pas stabilisé* :

    - Il contient des *pôles non stables*
    - Les trajectoires peuvent *diverger*
    - Le critère \( \Delta \theta(t) \to 0 \) *n’est pas satisfait durablement*
    """
    )
    return


@app.cell
def _(Ar, Br, np, plt, solve_ivp):
    # Initial state: only theta = pi/4
    X0 = np.array([0, 0, np.pi/4, 0])

    # Try several k3 and k4 values manually
    k3 = 10
    k4 = 6
    K = np.array([0, 0, k3, k4])  # only control θ and θ̇

    # Closed-loop dynamics: Ẋ = (A - B K) X
    A_cl = Ar - Br @ K.reshape(1, -1)

    # Time vector
    t_span1 = [0, 20]
    t_eval1 = np.linspace(t_span1[0], t_span1[1], 1000)

    # Solve ODE
    sol1 = solve_ivp(lambda t, x: A_cl @ x, t_span1, X0)
    x_t = sol1.y
    t = sol1.t

    # Compute control input: Δφ(t)
    phi_t = -K @ x_t

    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, x_t[2], label=r"$\Delta \theta(t)$")
    plt.axhline(np.pi/2, color='red', linestyle='--', label=r"$\pm \frac{\pi}{2}$")
    plt.axhline(-np.pi/2, color='red', linestyle='--')
    plt.xlabel("Time [s]")
    plt.ylabel("θ (rad)")
    plt.title("Tilt Angle Over Time")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t, phi_t, label=r"$\Delta \phi(t)$")
    plt.axhline(np.pi/2, color='red', linestyle='--', label=r"$\pm \frac{\pi}{2}$")
    plt.axhline(-np.pi/2, color='red', linestyle='--')
    plt.xlabel("Time [s]")
    plt.ylabel("ϕ (rad)")
    plt.title("Control Input Over Time")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    return


app._unparsable_cell(
    r"""
    To solve:  
    \(\ddot{x} = - \frac{g \sqrt{2}}{2}\)

    We integrate twice while respecting: \(x(0) = 0\), \(\dot{x}(0) = 0\).

    \(\dot{x}(t) = - \frac{g \sqrt{2}}{2} t\)

    \(x(t) = - \frac{g \sqrt{2}}{4} t^{2}\)
    """,
    name="_"
)


@app.cell
def _(g, np, plt):
    t7 = np.linspace(0, 5, 500)
    x = - (g * np.sqrt(2) / 4) * t7**2

    plt.figure(figsize=(8, 5))
    plt.plot(t7, x, label=r"$x(t) = -\frac{g\sqrt{2}}{4} t^2$", color="purple")
    plt.xlabel("Temps (s)")
    plt.ylabel("x(t) [m]")
    plt.title("Trajectoire x(t) sous l'effet de la gravité")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Pole Assignment

    Using pole assignement, find a matrix

    $$
    K_{pp} =
    \begin{bmatrix}
    ? & ? & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K_{pp} \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    satisfies the conditions defined for the manually tuned controller and additionally:

      - result in an asymptotically stable closed-loop dynamics,

      - make $\Delta x(t) \to 0$ in approximately $20$ sec (or less).

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ⭐ Answer


    We design a full-state feedback controller:

    \[
    \Delta \phi(t) = -K_{pp} \cdot 
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix}
    \]

    with:

    \[
    K_{pp} = [k_1, k_2, k_3, k_4]
    \]


    We want the closed-loop system to:

    - Be *asymptotically stable*

    - Bring \( \Delta x(t) \to 0 \) and \( \Delta \theta(t) \to 0 \)

    - Converge in *about 20 seconds* → dominant poles with real part ≈ -0.2 or faster

    - Use *pole placement* to assign eigenvalues of \( A_{cl} = A - B K \)

    ## Final Answer: Pole Placement Controller

    We chose the poles:

    \[
    \text{{poles}} = [-0.3, -0.4, -0.5, -0.6]
    \]

    The resulting gain is:

    \[
    K_{{pp}} = \begin{bmatrix}
    \boxed{{{K_{pp[0]}:.2f}}} &
    \boxed{{{K_{pp[1]}:.2f}}} &
    \boxed{{{K_{pp[2]}:.2f}}} &
    \boxed{{{K_{pp[3]}:.2f}}}
    \end{bmatrix}
    \]

     The closed-loop dynamics are:

    - *Asymptotically stable* (eigenvalues in left half-plane)

    - *All states converge to 0*, including \( \Delta x(t) \)

    - *Constraints* \( |\Delta \theta(t)| < \pi/2 \), \( |\Delta \phi(t)| < \pi/2 \) are respected
    """
    )
    return


@app.cell
def _(a, g, np, plt, solve_ivp):

    from scipy.signal import place_poles



    A9 = np.array([
        [0, 1, 0, 0],
        [0, 0, -g, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    B9 = np.array([
        [0],
        [-g],
        [0],
        [-a]
    ])

    # Desired pole locations (all negative real parts, stable, moderately fast)
    # Convergence in ~20 sec → real parts around -0.2 to -0.4
    desired_poles = np.array([-0.3, -0.4, -0.5, -0.6])

    # Use scipy's pole placement
    controller = place_poles(A9, B9, desired_poles)
    K_pp = controller.gain_matrix.flatten()

    print("Pole placement gain K_pp =", K_pp)

    # Closed-loop system
    A_cl9 = A9 - B9 @ K_pp.reshape(1, -1)

    # Initial state: theta = pi/4
    X09 = np.array([0, 0, np.pi/4, 0])
    t_span = [0, 20]
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    sol9 = solve_ivp(lambda t, x: A_cl9 @ x, t_span, X09)

    # Output
    x_t9 = sol9.y
    t9 = sol9.t
    phi_t9 = -K_pp @ x_t9

    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t9, x_t9[0], label=r"$\Delta x(t)$")
    plt.plot(t9, x_t9[2], label=r"$\Delta \theta(t)$")
    plt.axhline(np.pi/2, color='red', linestyle='--', label=r"$\pm \pi/2$")
    plt.axhline(-np.pi/2, color='red', linestyle='--')
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.title("State Evolution")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t9, phi_t9, label=r"$\Delta \phi(t)$")
    plt.axhline(np.pi/2, color='red', linestyle='--', label=r"$\pm \pi/2$")
    plt.axhline(-np.pi/2, color='red', linestyle='--')
    plt.xlabel("Time [s]")
    plt.title("Control Input")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Optimal Control

    Using optimal, find a gain matrix $K_{oc}$ that satisfies the same set of requirements that the one defined using pole placement.

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ⭐ Answer

    We want to find an *optimal gain*:

    \[
    K_{oc} \in \mathbb{R}^{1 \times 4}
    \]

    using the *Linear Quadratic Regulator (LQR)* method, which minimizes the cost function:

    \[
    J = \int_0^\infty \left( X^\top Q X + u^\top R u \right) dt
    \]

    ---

    ## Design Insight:

    - Matrix \( Q \) penalizes state errors (e.g. position, tilt)
    - Matrix \( R \) penalizes control effort \( \Delta \phi \)
    - To achieve convergence in ~20 sec:
      - Set *moderate weights* on \( \Delta x, \Delta \theta \)
      - Keep \( R \) small enough so that control is responsive but bounded


    ## Final Answer: LQR Optimal Controller

    We chose:

    - \( Q = \text{{diag}}(5, 0.1, 20, 1) \): prioritizes reducing lateral position and tilt
    - \( R = 0.05 \): balances control effort
    - Initial tilt: \( \Delta \theta(0) = \frac{{\pi}}{{4}} \)

    The resulting gain is:

    \[
    K_{{oc}} = \begin{bmatrix}
    \boxed{{{K_oc[0]:.2f}}} &
    \boxed{{{K_oc[1]:.2f}}} &
    \boxed{{{K_oc[2]:.2f}}} &
    \boxed{{{K_oc[3]:.2f}}}
    \end{bmatrix}
    \]

    Performance:

    - Closed-loop system is *asymptotically stable*
    - \( \Delta x(t), \Delta \theta(t) \to 0 \) in ~10–15 sec
    - \( |\Delta \theta(t)| < \frac{\pi}{2} \), \( |\Delta \phi(t)| < \frac{\pi}{2} \)

    All design requirements satisfied using optimal LQR control!
    """
    )
    return


@app.cell
def _(a, g, np, plt, solve_ivp):

    from scipy.linalg import solve_continuous_are


    A11 = np.array([
        [0, 1, 0, 0],
        [0, 0, -g, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    B11 = np.array([
        [0],
        [-g],
        [0],
        [-a]
    ])

    # Cost matrices
    Q11 = np.diag([5, 0.1, 20, 1])  # Penalize x and θ more
    R11 = np.array([[0.05]])        # Moderate control effort penalty

    # Solve Continuous Algebraic Riccati Equation
    P11 = solve_continuous_are(A11, B11, Q11, R11)
    K_oc11 = np.linalg.inv(R11) @ B11.T @ P11
    K_oc11 = K_oc11.flatten()

    print("LQR gain K_oc =", K_oc11)

    # Closed-loop dynamics
    A_cl11 = A11 - B11 @ K_oc11.reshape(1, -1)
    X011 = np.array([0, 0, np.pi/4, 0])
    t_span11 = [0, 20]
    t_eval11 = np.linspace(t_span11[0], t_span11[1], 1000)
    sol11 = solve_ivp(lambda t, x: A_cl11 @ x, t_span11, X011)

    x_t11 = sol11.y
    t11 = sol11.t
    phi_t11 = -K_oc11 @ x_t11

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t11, x_t11[0], label=r"$\Delta x(t)$")
    plt.plot(t11, x_t11[2], label=r"$\Delta \theta(t)$")
    plt.axhline(np.pi/2, color='red', linestyle='--', label=r"$\pm \pi/2$")
    plt.axhline(-np.pi/2, color='red', linestyle='--')
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.title("State Evolution")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t11, phi_t11, label=r"$\Delta \phi(t)$")
    plt.axhline(np.pi/2, color='red', linestyle='--')
    plt.axhline(-np.pi/2, color='red', linestyle='--')
    plt.xlabel("Time [s]")
    plt.title("Control Input")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Validation

    Test the two control strategies (pole placement and optimal control) on the "true" (nonlinear) model and check that they achieve their goal. Otherwise, go back to the drawing board and tweak the design parameters until they do!
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
