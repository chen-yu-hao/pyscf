#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

'''Reproduce the frozen-core C-atom UHF/cc-pVDZ quadruples benchmark.

Eriksen et al., JCP 144, 194103 (2016), DOI:10.1063/1.4948781,
SI Tables S2/S5 report -26.91 percentage points of quadruples recovery
error and +0.005 kcal/mol relative to CCSDTQ. With four active electrons,
FCI in exactly the same frozen-core UHF Hamiltonian is the CCSDTQ limit.
Compare at the printed precision, not at unreported hartree precision.
'''

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh
import pyscf
from pyscf import cc, fci, gto, lib, scf
from pyscf.cc import uccsdt_q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lib.num_threads(4)
    mol = gto.M(atom='C 0 0 0', basis='cc-pvdz', spin=2, charge=0,
                cart=False, verbose=3)
    mf = scf.UHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-10, max_cycle=200)
    assert mf.converged
    _, _, stable, _ = mf.stability(internal=True, external=False, return_status=True)
    mycc = cc.UCCSDT(mf, frozen=1, compact_tamps=True)
    mycc.conv_tol = 1e-14
    mycc.conv_tol_normt = 1e-12
    mycc.max_cycle = 200
    mycc.kernel()
    assert mycc.converged
    eris = mycc.ao2mo()
    start = time.perf_counter()
    bracket, paren = mycc.ccsdt_q(eris=eris)
    correction_seconds = time.perf_counter()-start

    # Construct the effective Hamiltonian with the very same frozen alpha
    # and beta core orbitals, including their constant energy and mean field.
    dmcore = np.asarray([c[:, :1] @ c[:, :1].T for c in mf.mo_coeff])
    hcore = mf.get_hcore()
    vcore = mf.get_veff(mol, dmcore)
    ecore = (mol.energy_nuc() + np.einsum('ij,ji', hcore, dmcore.sum(axis=0)) +
             .5*np.einsum('sij,sji', vcore, dmcore))
    active = eris.mo_coeff
    h1 = tuple(c.T @ (hcore+v) @ c for c, v in zip(active, vcore))
    eri = tuple(np.ascontiguousarray(v.transpose(0, 2, 1, 3))
                for v in (eris.pppp, eris.pPpP, eris.PPPP))
    norb = mycc.nmo[0]
    assert norb == mycc.nmo[1] and sum(mycc.nocc) == 4
    ci_hf = np.zeros(tuple(math.comb(norb, n) for n in mycc.nocc))
    ci_hf[0, 0] = 1.
    reconstructed_hf = fci.direct_uhf.energy(h1, eri, ci_hf, norb, mycc.nocc) + ecore
    np.testing.assert_allclose(reconstructed_hf, mf.e_tot, atol=1e-10, rtol=0)
    solver = fci.direct_uhf.FCISolver(mol)
    start = time.perf_counter()
    # This fixed four-electron problem has only 3718 determinants. Diagonalize
    # its full Hamiltonian to remove Davidson stopping-tolerance uncertainty.
    addresses, hamiltonian = solver.pspace(h1, eri, norb, mycc.nocc, np=ci_hf.size)
    eigenvalues, eigenvectors = eigh(hamiltonian, subset_by_index=(0, 2), overwrite_a=True)
    exact = eigenvalues[0] + ecore
    vector = np.empty(ci_hf.size)
    vector[addresses] = eigenvectors[:, 0]
    vector = vector.reshape(ci_hf.shape)
    fci_seconds = time.perf_counter()-start
    h2 = solver.absorb_h1e(h1, eri, norb, mycc.nocc, .5)
    residual = solver.contract_2e(h2, vector, norb, mycc.nocc) - (exact-ecore)*vector
    fci_converged = np.linalg.norm(residual) < 1e-10
    exact_quadruples = exact-mycc.e_tot
    recovery_error = 100*(paren/exact_quadruples-1)
    error_kcal = (mycc.e_tot+paren-exact)*627.5094740631
    match_recovery = abs(recovery_error-(-26.91)) <= .005
    match_energy = abs(error_kcal-.005) <= .0005
    root = Path(__file__).resolve().parents[2]
    native = Path(uccsdt_q._native_library()._name)
    report = {
        'system': 'C atom, neutral triplet ground state', 'basis': mol.basis,
        'cartesian': mol.cart, 'frozen': 1, 'nocc': list(mycc.nocc),
        'nmo': list(mycc.nmo), 'pyscf': pyscf.__version__, 'threads': lib.num_threads(),
        'scf_converged': bool(mf.converged), 'ccsdt_converged': bool(mycc.converged),
        'fci_converged': bool(fci_converged), 'uhf_spin_square': mf.spin_square()[0],
        'fci_solver': 'full determinant-space diagonalization', 'fci_determinants': ci_hf.size,
        'lowest_fci_states_eh': (eigenvalues+ecore).tolist(),
        'fci_residual_norm': float(np.linalg.norm(residual)),
        'fci_hf_determinant_weight': float(abs(vector[0, 0])**2),
        'uhf_internally_stable': bool(stable),
        'scf_gradient_norm': float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ))),
        'convergence': {'scf_energy': mf.conv_tol, 'scf_gradient': mf.conv_tol_grad,
                        'cc_energy': mycc.conv_tol, 'cc_amplitudes': mycc.conv_tol_normt,
                        'fci_residual': 1e-10},
        'scf_energy_eh': mf.e_tot, 'reconstructed_hf_energy_eh': float(reconstructed_hf),
        'ccsdt_energy_eh': mycc.e_tot, 'q_bracket_eh': bracket, 'q_paren_eh': paren,
        'ccsdt_q_energy_eh': mycc.e_tot+paren, 'frozen_core_fci_energy_eh': float(exact),
        'exact_quadruples_eh': float(exact_quadruples),
        'recovery_error_percentage_points': float(recovery_error),
        'ccsdt_q_minus_ccsdtq_kcal_mol': float(error_kcal),
        'literature': {'doi': '10.1063/1.4948781', 'si_table_S2_percentage_points': -26.91,
                       'si_table_S5_kcal_mol': .005,
                       'matches_S2_printed_precision': bool(match_recovery),
                       'matches_S5_printed_precision': bool(match_energy)},
        'correction_seconds': correction_seconds, 'fci_seconds': fci_seconds,
        'source_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                          for name in ('pyscf/cc/uccsdt_q.py', 'pyscf/lib/ccsdt/uccsdt_q.c',
                                       'examples/cc/70-validate_uccsdt_q_literature.py')},
        'native_sha256': hashlib.sha256(native.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    assert fci_converged, 'FCI eigenvector residual failed; see the saved report.'
    assert match_recovery and match_energy, 'Literature comparison failed; see the saved report.'


if __name__ == '__main__':
    main()
