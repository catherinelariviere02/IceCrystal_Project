import freud
import gsd.hoomd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import plotly.graph_objects as go

def bod(box, points, file, r_max, dir): 
    #create bond order diagram object
        n_bins_theta = 200
        n_bins_phi = 200
        bod = freud.environment.BondOrder((n_bins_theta, n_bins_phi))

        #create arrays for plotting
        phi = np.linspace(0, np.pi, n_bins_phi)
        theta = np.linspace(0, 2 * np.pi, n_bins_theta)
        phi, theta = np.meshgrid(phi, theta)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        #compute bond order diagram
        bod_array = bod.compute(
            system=(box, points), neighbors={"r_max": r_max, "r_min": 0}
        ).bond_order

        #clean up polar bins for plotting
        bod_array = np.clip(bod_array, 0, np.percentile(bod_array, 99))

        fig = go.Figure(data=
                        [go.Surface(x=x,y=y,z=z,surfacecolor=bod_array / np.max(bod_array), showscale=False)])
        fig.update_traces(lighting_ambient=1, colorscale="Viridis", selector=dict(type="surface"))
        fig.update_layout(
            scene = dict(xaxis = dict(visible=False), yaxis = dict(visible=False),zaxis = dict(visible=False)),
            title=dict(text='File: '+file), autosize=False,
            width=500, height=500,
            margin=dict(l=0, r=0, b=0, t=40)
            )
        fig.write_html(dir + f"finalframe_bod_rm{str(r_max).replace(".", "p")}.html")
        fig.show()

def rmsd(dir, data):
    num_frames = len(data)
    N = data[0].particles.N
    positions = np.zeros((num_frames, N, 3), dtype = float)

    for i, snap in enumerate(data): 
        box = freud.box.Box.from_box(snap.configuration.box)
        positions[i] = box.unwrap(snap.particles.position, snap.particles.image)

    msd = freud.msd.MSD(box, mode = "direct")
    msd.compute(positions)
    msd.plot()
    plt.savefig(dir + f"rmsd.pdf")

dir = "../../data/cubic/"
file = "trajectory_temp.gsd"

data = gsd.hoomd.open(dir + file, 'r')

print(len(data))
box, points = data[-1].configuration.box, data[-1].particles.position

#calculate and plot bod
for r_max in [1.3, 1.5, 1.6, 2, 2.2]:
    bod(box, points, file, r_max = r_max, dir = dir)

#rmsd(dir, data)
