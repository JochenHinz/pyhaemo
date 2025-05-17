from artery import Artery
from plot import plot_frames
from aux import centerline_length


def plot_swc_data():
  filename = "data/swc/BH0004.CNG.swc"
  artery = Artery.from_swc(filename)
  artery = artery.create_proximity_tree(eps=1e-3)
  # import ipdb
  # ipdb.set_trace()
  # artery.plot_pyvista()
  artery.plot_mesh(dx=0.75)


def plot_csv_data():
  artery = Artery.from_csv("data/csv/brain") \
                 .chordlength_parameterize() \
                 .normalize() \
                 .create_proximity_tree(0.05)

  artery = artery.bifcut()
  artery = artery.biftruncate(rfac=2)

  # plot_frames(artery)

  from connectivity import biftruncate
  for bif in artery.nontrivial_bifs:
    artery = biftruncate(artery, bif, distances=0.1)

  from mesh import make_mesh, plot_mesh

  artery = artery.orient()
  # plot_frames(artery)

  vessels, bifs = make_mesh(artery)
  lengths = [centerline_length(cline) for cline in artery.centerlines]
  dx = 0.02
  nsteps_z = [max(int(length / dx), 2) for length in lengths]
  plot_mesh(vessels, bifs, nsteps_z, npoints_cross=5)

  # artery.biftruncate().plot_mesh()


if __name__ == "__main__":
  # plot_swc_data()
  plot_csv_data()
