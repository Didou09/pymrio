""" NTNU specific tools for using pymrio
"""


import os
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.io

from pymrio.core.mriosystem import IOSystem
from pymrio.core.mriosystem import Extension
from pymrio.tools.iometadata import MRIOMetaData

# Constants and global variables
from pymrio.core.constants import PYMRIO_PATH


# NTNU mat structure info,
# this defines the strings of range in the mat structure
MAT_STRUCTURE = {
    # toplevel division
    'data': 'IO',
    'meta': 'meta',

    # Matrix names
    'Z': 'Z',           # flow matrix
    'Y': 'Y',           # final demand matrix
    'F': 'F',           # factor inputs
    'F_hh': 'F_hh',     # household inputs

    # monetary core
    'mon_labels': 'labsZ',
    'mon_sec_slice': np.index_exp[:, 1],
    'mon_reg_slice': np.index_exp[:, 5],
    'mon_unit': 'unit',

    # final demand
    'fd_labels': 'labsY',
    'fd_cat_slice': np.index_exp[:, 1],
    'fd_reg_slice': np.index_exp[:, 4],

    # stressor list
        # The stressor unit/compartment is not consistent in the
        # meta structure - comp/unit here are just placeholders to read
        # all data,  which gets sorted out afterwards.
    'stress_labels': 'labsF',
    'stress_name_slice': np.index_exp[:, 0],
    'stress_comp_slice': np.index_exp[:, 1],
    'stress_unit_slice': np.index_exp[:, 2],

    # other meta data
    'version': 'ver',
    'year': 'years',
    
    # settings
    'empty_comp': 'n/a',
}


def parse_mat(mat_file, name, io_system, struct=MAT_STRUCTURE):
    """Converts the mat file to a pymrio object.

    This function works with the Z (transaction) matrix version of the
    mat files. All matrices can either be sparse or dense (checked within
    the function).

    All satellite accounts (the F and F_hh) data is included as one
    single satellite account. Value added is not parsed (included in the
    single F).

    Note
    ----

    Characterization matrix is not included yet.

    Parameters
    ----------
    mat_file : str or pathlib.Path
        File with path to the mat file. The function currently works with
        Z (transaction) matrices.
    name : str
        How the pymrio object should be named.
    io_system : str
        pxp or ixi (currently not in the mat meta data).
    struct: dict, optional
        Structure of the underlying mat file.
        See the dict ntnu.MAT_STRUCTURE for an example. 
        This is used as default.

    Returns
    -------
    pymrio object

    """
    mat_file = os.path.normpath(str(mat_file))

    logging.debug('PARSER: Read matlab data from {}'.format(mat_file))
    mat_all = scipy.io.loadmat(
        mat_file,
        struct_as_record=False,
        squeeze_me=True)
    mat_io = mat_all[struct['data']]
    mat_meta = mat_all[struct['meta']]

    ll_sec = mat_meta.__dict__[
        struct['mon_labels']][struct['mon_sec_slice']].tolist()
    ll_reg = mat_meta.__dict__[
        struct['mon_labels']][struct['mon_reg_slice']].tolist()

    ll_fd_cat = mat_meta.__dict__[
        struct['fd_labels']][struct['fd_cat_slice']].tolist()
    ll_fd_reg = mat_meta.__dict__[
        struct['fd_labels']][struct['fd_reg_slice']].tolist()

    ll_stress_names = mat_meta.__dict__[
        struct['stress_labels']
        ][struct['stress_name_slice']].tolist()
    _ll_stress_comp = mat_meta.__dict__[
        struct['stress_labels']
        ][struct['stress_comp_slice']].tolist()
    _ll_stress_unit = mat_meta.__dict__[
        struct['stress_labels']
        ][struct['stress_unit_slice']].tolist()
    ll_stress_comp = [
        struct['empty_comp'] if len(unit) == 0 else orig_comp
        for unit, orig_comp in zip(_ll_stress_unit, _ll_stress_comp)]
    ll_stress_unit = [
        unit_orig if type(unit_orig) is str else unit_in_comp
        for unit_orig, unit_in_comp in zip(_ll_stress_unit, _ll_stress_comp)]

    multiind_mon = pd.MultiIndex.from_arrays(
        [ll_reg, ll_sec],
        names=[u'region', u'sector'])
    multiind_fd_col = pd.MultiIndex.from_arrays(
        [ll_fd_reg, ll_fd_cat],
        names=[u'region', u'category'])
    multiind_stress_row = pd.MultiIndex.from_arrays(
        [ll_stress_names, ll_stress_comp],
        names=[u'stressor', u'compartment'])

    ss_version = mat_meta.__dict__[struct['version']]
    ss_year = mat_meta.__dict__[struct['year']]
    ss_io_system = io_system  # no in the meta data - passed as parameter

    _Z = mat_io.__dict__[struct['Z']]
    _Z = _Z.todense() if scipy.sparse.issparse(_Z) else _Z
    df_Z = pd.DataFrame(
            data=_Z,
            index=multiind_mon,
            columns=multiind_mon,
            )
    del _Z

    _Y = mat_io.__dict__[struct['Y']]
    _Y = _Y.todense() if scipy.sparse.issparse(_Y) else _Y
    df_Y = pd.DataFrame(
            data=_Y,
            index=multiind_mon,
            columns=multiind_fd_col,
            )
    del _Y

    _F = mat_io.__dict__[struct['F']]
    _F = _F.todense() if scipy.sparse.issparse(_F) else _F
    df_F = pd.DataFrame(
            data=_F,
            index=multiind_stress_row,
            columns=multiind_mon,
            )
    del _F

    _F_hh = mat_io.__dict__[struct['F_hh']]
    _F_hh = _F_hh.todense() if scipy.sparse.issparse(_F_hh) else _F_hh
    df_F_hh = pd.DataFrame(
            data=_F_hh,
            index=multiind_stress_row,
            columns=multiind_fd_col,
            )
    del _F_hh

    df_mon_unit = pd.DataFrame(
            data=mat_meta.__dict__[struct['mon_unit']],
            index=multiind_mon,
            columns=['unit'],
            )
    df_F_unit = pd.DataFrame(
            data=ll_stress_unit,
            index=multiind_stress_row,
            columns=['unit'],
            )

    logging.debug('PARSER: Build pymrio system')
    io_to_py = IOSystem()
    io_to_py.name = name
    io_to_py.version = ss_version
    io_to_py.year = ss_year
    io_to_py.iosystem = ss_io_system

    logging.debug('PARSER: Build extension system')
    io_to_py.Z = df_Z
    io_to_py.Y = df_Y
    io_to_py.unit = df_mon_unit
    io_to_py.satellite = Extension(
        name='Satellite Accounts',
        F=df_F,
        F_hh=df_F_hh,
        unit=df_F_unit)

    meta_rec = MRIOMetaData(
        location=os.path.dirname(mat_file),
        description='MRIO parsed from NTNU mat structure',
        name=name,
        system=ss_io_system,
        version=ss_version,
    )
    meta_rec._add_fileio('MAT MRIO structure parsed from {}'.
                         format(mat_file))
    return io_to_py
