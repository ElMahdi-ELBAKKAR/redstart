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


if __name__ == "__main__":
    app.run()
