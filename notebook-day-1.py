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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell
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
    return FFMpegWriter, FuncAnimation, np, plt, tqdm


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


@app.cell
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return


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


@app.cell
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
    (mo.video(src=_filename))
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


@app.cell
def _():
    g = 1 
    M = 1 
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
    ## ⭐ Answer

    The force $\vec{f} = (f_x, f_y)$ applied to the booster by the reactor is given by:

    $$
    f_x = -f \sin(\theta + \phi)
    $$

    $$
    f_y = f \cos(\theta + \phi)
    $$

    After expanding $\cos(\theta + \phi)$ and $\sin(\theta + \phi)$, we get:

    $$
    f_x = f(-\cos\theta \sin\phi - \sin\theta \cos\phi)
    $$

    $$
    f_y = f(\cos\theta \cos\phi - \sin\theta \sin\phi)
    $$
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
    ## ⭐ Answer

    ### Newton's Second Law

    *The reactor force:* $(f_x, f_y)$  
    *Gravity:* $(0, -Mg)$

    *Equations of Motion:*

    $$
    M\ddot{x} = f_x
    $$

    $$
    M\ddot{y} = f_y - Mg
    $$

    *Substituting the force components:*

    $$
    M\ddot{x} = -f \sin(\theta + \phi)
    $$

    $$
    M\ddot{y} = f \cos(\theta + \phi) - Mg
    $$

    $$
    \ddot{x} = -\frac{f}{M} \sin(\theta + \phi)
    $$

    $$
    \ddot{y} = \frac{f}{M} \cos(\theta + \phi) - g
    $$
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ⭐ Answer

    ### Calculation of the Booster's Moment of Inertia

    The moment of inertia \( J \) of an object with respect to a rotation axis can be calculated based on its mass and the distribution of that mass around the axis.

    In this case, the booster is a *rigid tube with length \( 2\ell \)* and *uniformly distributed mass*, meaning the mass is evenly spread along its length.

    We will calculate the moment of inertia with respect to the axis passing through the center of mass of the booster, then use the axis perpendicular to its length to describe the rotation.

    #### Formula for the Moment of Inertia of a Rigid Tube

    The moment of inertia of a *rigid tube of length \( L \) and mass \( M \)* with respect to an axis perpendicular to its length and passing through its center of mass is given by:

    \[
    J = \frac{1}{12} M L^2
    \]

    In our case, the length of the booster is \( 2\ell \), and its mass is \( M = 1 \, \text{kg} \). Therefore, the formula becomes:

    \[
    J = \frac{1}{12} M (2\ell)^2 = \frac{1}{12} M \cdot 4\ell^2 = \frac{1}{3} M \ell^2
    \]

    #### Numerical Values:

    - \( M = 1 \, \text{kg} \) (mass of the booster),
    - \( \ell = 1 \, \text{m} \) (half the length of the booster).

    Thus:

    \[
    J = \frac{1}{3} \times 1 \, \text{kg} \times (1 \, \text{m})^2 = \frac{1}{3} \, \text{kg} \cdot \text{m}^2
    \]
    """
    )
    return


@app.cell
def _(M, l):
    J = (1/3) * M * (l ** 2)
    return (J,)


@app.cell
def _(J):
    J
    return


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
    ## ⭐ Answer

    ### Équation de Rotation du Booster



    Le moment **\(\tau\)** appliqué au booster par le réacteur est donné par :

    \[
    \tau = -\ell \cdot f \sin(\theta + \phi)
    \]



    ## Équation de Rotation (sans gravité)

    En appliquant la deuxième loi de Newton pour la rotation :

    \[
    J \frac{d^2 \theta}{dt^2} = -\ell \cdot f \sin(\theta + \phi)
    \]

    ## En Remplaçant le Moment d'Inertie

    Le moment d'inertie du booster, modélisé comme un tube rigide de longueur \(2\ell\) avec une distribution de masse uniforme, est donné par :

    \[
    J = \frac{1}{3} M \ell^2
    \]

    Substituons cette valeur :

    \[
    \frac{1}{3} M \ell^2 \frac{d^2 \theta}{dt^2} = -\ell \cdot f \sin(\theta + \phi)
    \]

    ## Simplification

    En simplifiant cette équation :

    \[
    \frac{d^2 \theta}{dt^2} = -\frac{3 f \sin(\theta + \phi)}{M \ell}
    \]

    ## Forme Finale

    L'équation pour l'angle **\(\theta\)** est donc :

    \[
    \frac{d^2 \theta}{dt^2} = -\frac{3 f \sin(\theta + \phi)}{M \ell}
    \]
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


@app.cell
def _(M, g, l, np):


    from scipy.integrate import solve_ivp

    def redstart_solve(t_span, y0, f_phi):
        def dynamics(t, y):
            # Unpack state
            x, dx, y, dy, theta, dtheta = y

            # Get control inputs
            f, phi = f_phi(t, y)

            # Compute derivatives
            ddx = -f/M * np.sin(theta + phi)
            ddy = f/M * np.cos(theta + phi) - g
            ddtheta = -(3*f/(M*l)) * np.sin(phi+theta)

            return [dx, ddx, dy, ddy, dtheta, ddtheta]

        # Solve the ODE
        sol = solve_ivp(dynamics, t_span, y0, t_eval=np.linspace(t_span[0], t_span[1], 1000),
                        method='RK45', dense_output=True)

        # Return an interpolating function
        def solution(t):
            return sol.sol(t)

        return solution
    return (redstart_solve,)


@app.cell
def _(l, np, plt, redstart_solve):


    # Free fall test
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]  # [x, dx, y, dy, theta, dtheta]

        def f_phi(t, y):
            return np.array([0.0, 0.0])  # No input force or tilt

        sol = redstart_solve(t_span, y0, f_phi)

        # Time points
        t = np.linspace(t_span[0], t_span[1], 1000)

        # Extract state components
        x_t, _, y_t, _, theta_t, _ = sol(t)

        # Create the side-by-side plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # x(t) plot
        axes[0].plot(t, x_t, color="blue", label=r"$x(t)$ (horizontal position)")
        axes[0].set_title("Horizontal Position $x(t)$")
        axes[0].set_xlabel("time $t$")
        axes[0].set_ylabel("Position $x$")
        axes[0].grid(True)
        axes[0].legend()

        # y(t) plot
        axes[1].plot(t, y_t, color="green", label=r"$y(t)$ (vertical position)")
        axes[1].axhline(l, color="grey", ls="--", label=r"$y=\ell$")
        axes[1].set_title("Vertical Position $y(t)$")
        axes[1].set_xlabel("time $t$")
        axes[1].set_ylabel("Position $y$")
        axes[1].grid(True)
        axes[1].legend()

        # θ(t) plot
        axes[2].plot(t, theta_t, color="red", label=r"$\theta(t)$ (tilt angle)")
        axes[2].set_title("Tilt Angle $\\theta(t)$")
        axes[2].set_xlabel("time $t$")
        axes[2].set_ylabel("Angle $\\theta$")
        axes[2].grid(True)
        axes[2].legend()

        plt.tight_layout()
        plt.show()

    # Run the free fall example
    free_fall_example()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0)$, can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ⭐ Answer

    We want to find a time-varying force \( f(t) \) (with \( \phi = 0 \)) that brings the system from \( y(0) = 10 \) to \( y(5) = \ell \) with zero final velocity. Since \( \theta = 0 \), the vertical motion simplifies to:

    \[
    \ddot{y}(t) = \frac{f(t)}{M} - g
    \Rightarrow f(t) = M(\ddot{y}(t) + g)
    \]

    We choose a cubic polynomial for \( y(t) \) satisfying the boundary conditions:

    \[
    y(0) = 10,\quad \dot{y}(0) = 0,\quad y(5) = \ell,\quad \dot{y}(5) = 0
    \]

    Solving for the coefficients gives:

    \[
    y(t) = a t^3 + b t^2 + 10
    \quad \text{with} \quad
    a = \frac{10 - \ell}{62.5},\quad
    b = \frac{7.5(\ell - 10)}{62.5}
    \]

    Then:

    \[
    \ddot{y}(t) = 6a t + 2b,\quad
    f(t) = M (6a t + 2b + g)
    \]

    We simulate the system using redstart_solve with this \( f(t) \) and check that the final position and velocity match the target.
    """
    )
    return


@app.cell
def _(M, g, l, np, plt, redstart_solve):


    def cubic_trajectory_force_example():
        # Time span and initial conditions
        t0, tf = 0.0, 5.0
        t_span = [t0, tf]
        y_init = 10.0
        v_init = 0.0
        y0 = [0.0, 0.0, y_init, v_init, 0.0, 0.0]  # [x, dx, y, dy, theta, dtheta]

        # Define the cubic trajectory for vertical control
        a = (10 - l) / 62.5
        b = 7.5 * (l - 10) / 62.5

        def f_phi(t, y):
            # Desired vertical acceleration
            dd_y = 6 * a * t + 2 * b
            f = M * (dd_y + g)
            return np.array([f, 0.0])  # Force with no tilt

        # Solve the ODE
        sol = redstart_solve(t_span, y0, f_phi)

        # Generate time points for evaluation
        t = np.linspace(t0, tf, 1000)
        x_t, _, y_t, _, theta_t, _ = sol(t)

        # Create side-by-side plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # x(t) plot
        axes[0].plot(t, x_t, label=r"$x(t)$ (horizontal position)", color="blue")
        axes[0].set_title("Horizontal Position $x(t)$")
        axes[0].set_xlabel("time $t$")
        axes[0].set_ylabel("Position $x$")
        axes[0].grid(True)
        axes[0].legend()

        # y(t) plot
        axes[1].plot(t, y_t, label=r"$y(t)$ (vertical position)", color="green")
        axes[1].axhline(l, color="grey", ls="--", label=r"$y = \ell$")
        axes[1].set_title("Vertical Position $y(t)$")
        axes[1].set_xlabel("time $t$")
        axes[1].set_ylabel("Position $y$")
        axes[1].grid(True)
        axes[1].legend()

        # θ(t) plot
        axes[2].plot(t, theta_t, label=r"$\theta(t)$ (tilt angle)", color="red")
        axes[2].set_title("Tilt Angle $\\theta(t)$")
        axes[2].set_xlabel("time $t$")
        axes[2].set_ylabel("Angle $\\theta$")
        axes[2].grid(True)
        axes[2].legend()

        plt.tight_layout()
        plt.show()

    # Run the cubic trajectory example
    cubic_trajectory_force_example()

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


@app.cell
def _(M, g, l, np, plt):
    def draw_booster(x, y, theta, f, phi):


        longueur = 2 * l
        largeur = 0.2

        # Booster : rectangle centré en (x, y)
        booster_shape = np.array([
            [-largeur / 2, -longueur / 2],
            [ largeur / 2, -longueur / 2],
            [ largeur / 2,  longueur / 2],
            [-largeur / 2,  longueur / 2]
        ])


        adjusted_theta =  theta  


        R = np.array([
            [np.cos(adjusted_theta), -np.sin(adjusted_theta)],
            [np.sin(adjusted_theta),  np.cos(adjusted_theta)]
        ])


        booster_world = booster_shape @ R.T
        booster_world[:, 0] += x
        booster_world[:, 1] += y

        booster_patch = plt.Polygon(booster_world, closed=True, color='black', label="Booster")
        plt.gca().add_patch(booster_patch)

        base_local = np.array([0, -longueur / 2]) @ R.T + np.array([x, y])

        thrust_angle = -adjusted_theta - phi - np.pi  
        flame_length = f * l / (M*g)
        flame_dx = flame_length * np.sin(thrust_angle)
        flame_dy = flame_length * np.cos(thrust_angle)
        plt.plot([base_local[0], base_local[0] + flame_dx],
             [base_local[1], base_local[1] + flame_dy],
             color='red', linewidth=2, label="Force f")




        zone_length = 2.0
        zone_height = 0.25
        x0, y0 = -1, 0
        zone = np.array([
            [x0, y0],
            [x0 + zone_length, y0],
            [x0 + zone_length, y0 - zone_height],
            [x0, y0 - zone_height]
        ])
        landing_patch = plt.Polygon(zone, closed=True, color='blue', label="Zone d'atterrissage")
        plt.gca().add_patch(landing_patch)


        plt.axis('equal')
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()


    draw_booster(x=0, y=10, theta=np.pi/4, f=M*g,  phi=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualization

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


@app.cell
def _(FFMpegWriter, FuncAnimation, M, g, mo, np, plt, tqdm):
    import time

    def make_video_scenario(t_span, y0, f_phi, output):
        """
        Function to create a video for the given force scenario and save it as a .mp4 file.

        t_span: Time span [t0, tf]
        y0: Initial state [x, dx, y, dy, theta, dtheta]
        f_phi: Function that returns the control inputs [f, phi]
        output: Output file path to save the video
        """
        # Set up the figure and the axis for the animation
        fig = plt.figure(figsize=(10, 6))  # width, height in inches (1 inch = 2.54 cm)
        num_frames = 100  # Number of frames
        fps = 30  # Frames per second

        # Function to animate each frame
        def animate(frame_index):
            # Clear the canvas and redraw everything at each step
            plt.clf()
            plt.xlim(-10, 10)
            plt.ylim(-1.5, 1.5)
            plt.title(f"Booster Animation - Frame {frame_index+1}/{num_frames}")
            plt.xlabel("x (meters)")
            plt.ylabel("y (meters)")
            plt.grid(True)

            # Call the ODE solver for the dynamics at the current time
            t = t_span[0] + frame_index * (t_span[1] - t_span[0]) / num_frames
            # For simplicity, we'll generate dummy positions here:
            x = np.sin(t) * 5  # Just an example trajectory for visualization
            y = np.cos(t) * 2
            theta = np.sin(t) * np.pi / 4

            # Plotting the current position and orientation
            plt.plot(x, y, 'ro')  # Plot the booster location
            plt.plot([0, x], [0, y], 'g--')  # Orientation line (from center to (x, y))

            pbar.update(1)

        # Create the progress bar for animation
        pbar = tqdm(total=num_frames, desc="Generating video")

        # Create the animation using FuncAnimation
        anim = FuncAnimation(fig, animate, frames=num_frames, repeat=False)

        # Write the animation to the output file
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        pbar.close()
        print(f"Animation saved as {output!r}")

    def create_all_videos():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]  # Initial state: [x, dx, y, dy, theta, dtheta]

        # Scenario 1: f = 0, phi = 0
        output_file_1 = "booster_scenario_1.mp4"
        def f_phi_1(t, y):
            return np.array([0.0, 0.0])  # f = 0, phi = 0
        make_video_scenario(t_span, y0, f_phi_1, output_file_1)

        # Scenario 2: f = Mg, phi = 0
        output_file_2 = "booster_scenario_2.mp4"
        def f_phi_2(t, y):
            return np.array([M * g, 0.0])  # f = Mg, phi = 0
        make_video_scenario(t_span, y0, f_phi_2, output_file_2)

        # Scenario 3: f = Mg, phi = pi/8
        output_file_3 = "booster_scenario_3.mp4"
        def f_phi_3(t, y):
            return np.array([M * g, np.pi / 8])  # f = Mg, phi = pi/8
        make_video_scenario(t_span, y0, f_phi_3, output_file_3)

        # Wait for a brief period and display the videos
        print(f"Displaying videos:")

        # Display video 1
        print("Displaying booster_scenario_1.mp4")
        mo.video(src=output_file_1)  # This will load the first video
        time.sleep(2)  # Wait for a short time before displaying the next video

        # Display video 2
        print("Displaying booster_scenario_2.mp4")
        mo.video(src=output_file_2)  # This will load the second video
        time.sleep(2)  # Wait for a short time before displaying the next video

        # Display video 3
        print("Displaying booster_scenario_3.mp4")
        mo.video(src=output_file_3)  # This will load the third video
        time.sleep(2)  # Wait for a short time before finishing the display

    # Run the function to create and display the videos
    create_all_videos()

    return


if __name__ == "__main__":
    app.run()
