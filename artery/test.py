from artery import Artery
from plot import plot_frames


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
                 .create_proximity_tree(0.15) \
                 .biftruncate(rfac=3) \
                 .orient()
  # artery = artery.take( list(chain(*artery.nontrivial_bifs[:2] )) )
  artery.plot_mesh()
  # import ipdb
  # ipdb.set_trace()

  # artery = artery.take(artery.nontrivial_bifs[5])
  plot_frames(artery)

  # artery.biftruncate().plot_mesh()


if __name__ == "__main__":
  # plot_swc_data()
  plot_csv_data()
