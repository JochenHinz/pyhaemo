from pathlib import Path


path_to_directory = Path(__file__).resolve().parent

OUTPUT_displacement_path = path_to_directory / 'displacement/'
OUTPUT_meshes_path = path_to_directory / 'meshes/'

VESSEL_DATA_PATHS = {
    'LAD': path_to_directory / 'data/reconSeptember/reconSept/reconstructions-LAD.npy',
    'CX': path_to_directory / 'data/reconSeptember/reconSept/reconstructions-CX.npy',
    'RCA': path_to_directory / 'data/reconSeptember/reconSept/reconstructions-RCA.npy',
}

# Acquisition frame rate, in frames per second, of the reconstructions --
# the same for every vessel.
ACQUISITION_FPS = 15

VESSEL_SAMPLING_RATES = {
    'LAD': 1 / ACQUISITION_FPS,
    'CX': 1 / ACQUISITION_FPS,
    'RCA': 1 / ACQUISITION_FPS,
}

# Vessel used by scripts (e.g. DataTime.py, TimeInterpolation.py) that load a
# single, fixed reconstruction rather than taking a vessel argument. Change
# this to point every one of those scripts at a different vessel at once.
INPUT_VESSEL = 'RCA'
INPUT_data_path = VESSEL_DATA_PATHS[INPUT_VESSEL]
INPUT_sampling_rate = VESSEL_SAMPLING_RATES[INPUT_VESSEL]
