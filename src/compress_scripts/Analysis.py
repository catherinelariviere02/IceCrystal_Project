import freud
import gsd.hoomd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import plotly.graph_objects as go
import os
import cv2
from natsort import natsorted

def rdf(job, box, points, prop, r_max, dir, bins, frame, ax = None, label = None):
    """Helper function for plotting RDFs."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_title(prop, fontsize=16)
    rdf = freud.density.RDF(bins, r_max)
    rdf.compute(system=(box, points), reset=False)

    if label is not None:
        ax.plot(rdf.bin_centers, getattr(rdf, prop), label=label)
        ax.legend()
    else:
        ax.plot(rdf.bin_centers, getattr(rdf, prop))
    plt.savefig(job.fn(f"rdf/rdf_{frame}.png"))
    return ax   

def bod(job, box, points, file, r_max, dir, bins, frame): 
    #create bond order diagram object
        n_bins_theta = job.sp.bins
        n_bins_phi = job.sp.bins
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
        fig.write_image(job.fn(f"bod/finalframe_bod_{frame}.png"))

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
    plt.savefig(dir + f"rmsd.png")


def video(job, prop):
    path = f"{prop}/"
    video_name = job.fn(dir + f"{prop}.mp4")
    images = [img for img in os.listdir(path) if img.endswith((".jpg", ".jpeg", ".png"))]
    
    images = natsorted(images)
    print(images)

    # Set frame from the first image
    frame = cv2.imread(os.path.join(path, images[0]))
    height, width, layers = frame.shape

    # Video writer to create .avi file
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
    # Appending images to video
    for image in images:
        video.write(cv2.imread(os.path.join(path, image)))

    # Release the video file
    video.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")



#calculate and plot bod
def analyze(job):
    if os.path.isdir(job.workspace + "rdf") == False:
        os.mkdir(job.workspace + "rdf")

    if os.path.isdir(job.workspace + "bod") == False:
        os.mkdir(job.workspace + "bod")

    file = "trajectory.gsd"
    data = gsd.hoomd.open(job.fn(file), 'r')

    for i in range(job.statepoint.logsteps):
        box, points = data[i].configuration.box, data[i].particles.position
        r_max = job.sp.rdf_rmax
        rdf(job, box, points, "rdf", r_max, dir, bins = 100, frame = i)

        r_max = job.sp.bod_rmax
        bod(job, box, points, file, r_max = r_max, dir = dir, bins = 500, frame = i)
        

    video("rdf")
    video("bod")
    #rmsd(dir, data)
