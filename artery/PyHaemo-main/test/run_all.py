import unittest
import sys


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
