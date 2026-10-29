"""Exact, compact NUMA wire payload; no occupation threshold is introduced."""

def _density_payload(dm):
    import numpy as np
    coeff = getattr(dm, 'mo_coeff', None)
    occ = getattr(dm, 'mo_occ', None)
    # Do not alter nonstandard tags; leave validation to the original worker.
    if (isinstance(coeff, np.ndarray) and isinstance(occ, np.ndarray)
            and coeff.ndim == 2 and occ.ndim == 1
            and coeff.shape == (dm.shape[0], occ.size)):
        keep = occ != 0
        # Keep negative/fractional occupations, and retain the all-zero case's
        # original shape for compatibility with alternate NumInt code paths.
        if keep.any() and not keep.all():
            coeff = np.ascontiguousarray(coeff[:, keep])
            occ = np.ascontiguousarray(occ[keep])
    return {'density': np.asarray(dm), 'mo_coeff': coeff, 'mo_occ': occ}


def send_payload(connections, dm):
    """Serialize once and send the standard Connection.recv-compatible frame."""
    from multiprocessing.reduction import ForkingPickler
    message = ForkingPickler.dumps(_density_payload(dm))
    for conn in connections:
        conn.send_bytes(message)
