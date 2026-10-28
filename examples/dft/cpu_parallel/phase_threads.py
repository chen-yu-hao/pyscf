"""Opt-in XC thread limit, independent of the SCF object's J/K thread count."""
from functools import wraps
from pyscf import lib


def limit_xc_threads(mf, threads=8):
    """Limit this SCF object's numerical integration; return the same object.

    Set the overall budget first with lib.num_threads, e.g. 24. The context
    restores it after each XC call, including when the integrator raises.
    No limit is applied to J/K or eigensolvers. Eight XC
    threads is a measured choice for this EPYC 7551 host, not a universal default.
    """
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError('XC threads must be a positive integer')
    ni = mf._numint
    for name in ('nr_rks', 'nr_uks'):
        original = getattr(ni, name)
        original = getattr(original, '_unlimited_xc', original)
        @wraps(original)
        def limited(*args, _original=original, **kwargs):
            with lib.with_omp_threads(min(threads, lib.num_threads())):
                return _original(*args, **kwargs)
        limited._unlimited_xc = original
        setattr(ni, name, limited)
    return mf
