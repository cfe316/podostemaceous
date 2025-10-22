# this file

def normalized_dx(mol, dx, n, T):
    """
    """
    mfp = mol.mean_free_path(n, T)
    return dx / mfp


def normalized_dt(mol, dt, n, T):
    """
    """
    mft = mol.mean_free_time(n, T)
    return dt / mft
