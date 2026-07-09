import unittest
import sys

# Force a non-interactive backend before anything else runs. Some tests
# (e.g. test_notwist.py) call matplotlib's plt.show() purely for visual
# debugging, with no bearing on their actual assertions. That's a harmless
# no-op under matplotlib's default backend -- but once PyQt6 has been
# imported anywhere in this process (the partery GUI tests do), matplotlib
# auto-detects it and switches to the interactive Qt backend, so plt.show()
# blocks forever waiting for a window to be closed that will never appear
# under a headless/offscreen display, hanging the whole run.
import matplotlib
matplotlib.use("Agg")


def run_all():
  # Discover all tests in the folder
  loader = unittest.TestLoader()
  suite = loader.discover("")

  runner = unittest.TextTestRunner(verbosity=2)
  result = runner.run(suite)

  # Optional: exit with appropriate status code
  sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
  run_all()
