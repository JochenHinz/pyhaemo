"""
Plot the reference displacement (unsigned magnitude, and signed along the
top/bottom axis) for LAD, CX and RCA, each using the maximum number of
heartbeats available in its reconstruction, and save the figures as vector
PDFs in the user's home directory.

    python3 run_reference_disp_plots.py
"""
from pathlib import Path

from pipeline import load_full_motion
from plots import plot_reference_disp, plot_signed_reference_disp

VESSELS = ['LAD', 'CX', 'RCA']


def main():
    for vessel in VESSELS:
        data = load_full_motion(vessel)

        save_path = Path.home() / f'{vessel}_reference_displacement.pdf'
        plot_reference_disp(
            data,
            title=f'{vessel} - reference displacement (full motion)',
            save_path=str(save_path),
        )
        print(f'saved {save_path}')

        signed_save_path = Path.home() / f'{vessel}_signed_reference_displacement.pdf'
        plot_signed_reference_disp(
            data,
            title=f'{vessel} - signed reference displacement (full motion)',
            save_path=str(signed_save_path),
        )
        print(f'saved {signed_save_path}')


if __name__ == '__main__':
    main()
