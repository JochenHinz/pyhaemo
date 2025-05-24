from ast import Tuple
from gc import callbacks
from artery import Artery
from pyhaemo.util import np, _, normalize
import pyvista as pv
import numpy as np
from pyhaemo.mesh import rectilinear, LineMesh, mesh_union
from preprocessing import split_edges_proximity
from aux import centerline_length, distance, sort_ccw
from functools import partial
from mesh import make_mesh, plot_mesh


user_bif_to_insert = []
user_bif_to_delete = []

#####pyvista callback#####
def create_bif(actor, plotter, bif):
    for name, act in plotter.renderer.actors.items():
        if act == actor and name.isdigit():
            bif.append(name)
            #position = plotter.pick_click_position()
            #print(position)
            print(f"Add vessel: {name} to the bifurcation")
            print(f"The bifurcation is composed by: {bif}")
            break
          
    print(f"Creating bifurcation with vessels: {bif}")
    user_bif_to_insert.append(bif)


def remove_and_store_label(actor, plotter):
    # Find the label (name) of the actor
    for name, act in plotter.renderer.actors.items():
        if act == actor:
            deleted_centerline.append(name)
            plotter.remove_actor(actor)
            print(f"Deleted centerline n_: {name}")
            break


def take_screenshot():
    """Callback to save a screenshot."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    plotter.screenshot(filename)
    print(f"Screenshot saved to {filename}")


def switch_to_create_bif():
    selected_bif = []
    plotter.disable_picking()  # Disable current picking observer
    plotter.enable_mesh_picking(
        callback=partial(create_bif, plotter=plotter, bif=selected_bif),
        use_actor=True, left_clicking=True, show_message=True
    )
    print("Switched to localize bifurcation")


def save_bifurcation():
    selected_bif = []
    plotter.disable_picking()  # Disable current picking observer
    plotter.enable_mesh_picking(
        callback=partial(create_bif, plotter=plotter, bif=selected_bif),
        use_actor=True, left_clicking=True, show_message=True
    )


def switch_to_remove_centerline():
    plotter.disable_picking()  # Disable current picking observer
    plotter.enable_mesh_picking(
        callback=partial(remove_and_store_label, plotter=plotter),
        use_actor=True, left_clicking=True, show_message=True
    )
    print("Switched to remove centerline")


def toggle_edges(state):
    """Callback to toggle mesh edge visibility based on checkbox state."""
    actor = plotter.actors['actor_name']  # Replace with your actor name
    actor.GetProperty().SetEdgeVisibility(state)
    print(f"Edges visibility set to: {state}")


# For a simple action like resetting the camera, a key binding is often more intuitive:
def reset_camera_view():
    """Callback to reset the camera to an isometric view."""
    plotter.reset_camera()
    plotter.view_isometric()
    print("Camera reset!")


def on_click(actor, position):
    for name, act in plotter.renderer.actors.items():
        if act == actor:
            deleted_centerline.append(name)
            print("Clicked on mesh at:", position)


def line_element(mesh: LineMesh):
  line_mesh = pv.PolyData()
  line_mesh.points = mesh.points
  line_mesh.lines = \
    np.concatenate([ np.full((len(mesh.pvelements), 1), mesh.nverts, dtype=int),
                              mesh.pvelements ], axis=1).ravel()
  return line_mesh


def create_artery_from_visual(artery: Artery, user_bif_to_insert) -> Artery:
  """
  Create an artery object from a visual representation.
  """
  npoints = 101
  centerlines = artery.centerlines


  for i, cline in enumerate(artery.centerlines, 1):
    mesh = rectilinear([np.linspace(*cline.knots[0][[0, -1]], npoints)])
    line = cline.sample_mesh(mesh)

    lmesh = line_element(line)
    plotter.add_mesh(lmesh, color="blue", line_width=3,  pickable=True, name=str(i))

    centerpoint = cline([np.mean(cline.knots[0][[0, -1]])]).ravel()
    plotter.add_points(centerpoint[_], color='red')
    plotter.add_point_labels(centerpoint[_], [str(i)])


  plotter.enable_mesh_picking(callback=partial(remove_and_store_label, plotter=plotter),
                              use_actor=True, left_clicking=False)

  # Add a checkbox button widget to toggle edges
  # The callback will receive a boolean 'state' (True/False)
  #plotter.add_checkbox_button_widget(
  #  toggle_edges,
  #  value=True,  # Initial state: edges are visible
  #  color_on='green',
  #  color_off='red',
  #  background_color='lightgray',
  #  position=(0.05, 0.9), # Position (x, y) as a fraction of the window size
  #  size=30 # Size of the button
  #)
# Add a light source
  plotter.add_light(pv.Light(position=(1, 1, 1), color='white'))

## You can also adjust the ambient and diffuse properties of the light
#  plotter.enable_anti_aliasing()
  #plotter.show_grid()
  #plotter.show_axes()
  plotter.set_background('white')  # You can also set a plain background if needed
# Enable higher-quality rendering settings
  #plotter.set_antialiasing('fast')
  plotter.enable_parallel_projection()

  # Assign key events to switch callbacks
  plotter.add_key_event("b", switch_to_create_bif)
  plotter.add_text("Press 'b' to enter in mode locate bifurcation, and select two branches with left click", position=(0, 0), font_size=12, color="black")


  plotter.add_key_event("d", switch_to_remove_centerline)
  plotter.add_text("Press 'd' to enter in mode Delete Branch, and select a branch with left click", position=(0, 35), font_size=12, color="black")

  plotter.add_key_event('r', reset_camera_view)
  plotter.add_text("Press 'r' to reset the view", position=(0, 105), font_size=12, color="black")

  plotter.add_key_event('s', take_screenshot)  # Press 's' to take screenshot
  plotter.add_text("Press 's' to take a screenshot", position=(0, 70), font_size=12, color="black")

  plotter.add_logo_widget('/home/fabio/logo.jpeg', position=[.8,.9])

  plotter.show()


  new_centerlines = [c for i,c in enumerate(centerlines) if str(i) not in deleted_centerline]
  new_radii = [r for i,r in enumerate(artery.radii) if str(i) not in deleted_centerline]

  new_bif = [[int(x) for x in inner] for inner in user_bif_to_insert]
  cleaned_bif = clean_duplicate_bif(new_bif)
  final_bif = clean_duplicate_values(cleaned_bif)

  import ipdb
  ipdb.set_trace()

  return Artery(new_centerlines, new_radii, final_bif)


def clean_duplicate_bif(bifs):
  seen = set()
  cleaned_bif = []

  for mybif in bifs:
    if tuple(mybif) not in seen:
      seen.add(tuple(mybif))
      cleaned_bif.append(mybif)

  return cleaned_bif


def clean_duplicate_values(bifs):
  cleaned_bif = []

  for mybif in bifs:
    seen = set()
    new_bif = []
    for myvalue in mybif:
      if myvalue not in seen:
        seen.add(myvalue)
        new_bif.append(myvalue)

    cleaned_bif.append(new_bif)

  return cleaned_bif




if __name__ == '__main__':

  plotter = pv.Plotter()
  deleted_centerline = []


  artery = Artery.from_csv("data/csv/aorta/")

  new_artery = create_artery_from_visual(artery, user_bif_to_insert)

  art = new_artery.chordlength_parameterize() \

  new_artery = split_edges_proximity(artery,0.3) \
           .reparameterize() \
           .biftruncate(rfac=2)


  print(artery.nontrivial_bifs)
  artery = artery.orient()
  lengths = [centerline_length(cline) for cline in artery.centerlines]
  dx = 0.05
  nsteps_z = [max(int(length / dx), 2) for length in lengths]
  print(nsteps_z)

  vessels, bifs = make_mesh(artery)
  plot_mesh(vessels, bifs, nsteps_z, npoints_cross=5)
