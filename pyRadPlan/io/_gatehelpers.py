from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from pyRadPlan.ct import CT


def write_ct_to_mhd(ct: CT, output_dir: Path, basename: str = "ct") -> Path:
    """
    Export a CT volume to an MetaImage (MHD/RAW) pair.

    Parameters
    ----------
    ct : CT
        The CT object to export.
    output_dir : Path
        Directory where the files will be written.
    basename : str
        Base name for the generated files (default: 'ct').

    Returns
    -------
    Path
        Path to the generated `.mhd` file.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    image = ct.cube_hu

    # SimpleITK expects filename without extension; SetFileName handles raw creation.
    mhd_path = output_dir / f"{basename}.mhd"

    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(mhd_path))
    writer.SetUseCompression(False)
    writer.Execute(image)

    return mhd_path


def write_mask_to_mhd(mask: sitk.Image, output_dir: Path, basename: str) -> Path:
    """
    Export a SimpleITK mask image to MHD/RAW.

    Parameters
    ----------
    mask : sitk.Image
        Binarized mask image.
    output_dir : Path
        Directory where the files will be written.
    basename : str
        Base name for the generated files.

    Returns
    -------
    Path
        Path to the generated `.mhd` file.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    mhd_path = output_dir / f"{basename}.mhd"
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(mhd_path))
    writer.SetUseCompression(False)
    writer.Execute(mask)

    return mhd_path


def numpy_to_mhd(array: np.ndarray, reference: sitk.Image, output_dir: Path, basename: str) -> Path:
    """
    Export a numpy array to MHD/RAW using spatial metadata from a reference image.

    Parameters
    ----------
    array : np.ndarray
        The data to export.
    reference : sitk.Image
        Spatial reference image providing origin, spacing, and direction.
    output_dir : Path
        Output directory.
    basename : str
        Base name for the generated files.

    Returns
    -------
    Path
        Path to the generated `.mhd` file.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)

    mhd_path = output_dir / f"{basename}.mhd"
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(mhd_path))
    writer.SetUseCompression(False)
    writer.Execute(image)

    return mhd_path
