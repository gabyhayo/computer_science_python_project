# +
import matplotlib.pyplot as plt
import h5py

def plot_defect(defect_id: int):
    plt.figure()
    with h5py.File("defects.hdf5", 'r') as fin:
        grp = fin[f"defect_{defect_id:04d}"]
        pos = grp['positions'][()]
        maxv = grp['maxpos'][()]
        minv = grp['minpos'][()]
        tri = grp['triangles'][()]


        ax = plt.subplot(projection='3d')
        ax.plot_trisurf(pos[:,0], pos[:, 1], pos[:, 2], triangles = tri,
                        color='orange', edgecolor='k', linewidth=0.25, alpha=1., shade=False)
        ax.title.set_text(f'defect {defect_id}')
        
        # Awful hack to keep the aspect of the defect.
        ax.plot([minv, maxv], [minv, maxv],[minv, maxv], alpha=0.)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
