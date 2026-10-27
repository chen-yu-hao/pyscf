# Kernel review guide for the complete implementation

The final source tree is identical to aggregate72e5b26c9. This guide identifies the equations, dispatch conditions and ownership rules a human author must understand before submitting the nine cumulative contributions. It is supporting review material, not a claim that human review has occurred.

1. **Grid grouping.** The offset box IDs form bounded nonnegative digits. A mixed-radix integer key has the same lexicographic order as the original three-column unique/inverse mapping. Stable sorting preserves order inside each box. Python integer arithmetic checks the total range before int64 operations; unsupported ranges retain the original implementation. Review `arg_group_grids` and the exact-index tests in `test_grids`.

2. **LibXC thread ownership.** Each thread owns one grid slice through density packing, functional-component evaluation and output assembly. Components are accumulated in their original order at each point. No two threads reduce into the same point. Check all component strides, spin layouts and empty slices when threads exceed points; do not generalize a supported derivative order to unsupported backend functionality. Review `libxc_itrf.c` and `test_libxc`.

3. **AO first derivatives.** For primitive j, the radial factor is `E_j(g)=fac*exp(-a_j*r(g)^2)`. Contracted values use `sum_j c_kj E_j`; first radial derivative factors use `sum_j(-2*a_j)c_kj E_j`. A short primitive-local buffer reuses exponential factors while retaining primitive order and screening. Polynomial derivatives and spherical/cartesian assembly continue through the original code. This does not rewrite higher AO derivative orders. Review `GTOcontract_exp1` and `test_grid_exp1`.

4. **Sparse DM density.** For real AO values phi and a real DM D:

   ```text
   rho = sum_mu,nu D_mu,nu phi_mu phi_nu
   grad rho = sum_mu,nu D_mu,nu [(grad phi_mu) phi_nu + phi_mu grad phi_nu]
   tau = 1/2 sum_mu,nu D_mu,nu grad phi_mu dot grad phi_nu
   ```

   The gradient becomes `2 phi^T D grad(phi)` only for the appropriate Hermitian case. The screened C kernels form local contractions and immediately accumulate outputs rather than retaining repeated full AO-by-DM intermediates. Preserve shell/bin/pair screening, DM orientation and the tau half-factor. Unsupported dtype, layout, alignment, Laplacian or nonsymmetric-gradient cases retain fallbacks. The same step bounds the block mask to the current block; dispatch must assess current sparsity rather than the unprocessed mask tail. Review `eval_rho1`, `VXCeval_rho_lda_gga_sparse`, `VXCeval_rho_mgga_sparse` and `test_numint_rho_fused`.

5. **MO density.** With `psi_i=sum_mu C_mu,i phi_mu`, evaluate `rho=sum_i occ_i psi_i^2`, `grad rho=2 sum_i occ_i psi_i grad psi_i`, and `tau=1/2 sum_i occ_i |grad psi_i|^2`. Packing `sqrt(abs(occ_i))*C_i` still requires the occupation sign in all outputs. The streamed kernel immediately contracts short grid blocks and may compress an AO column only when its value and all three first derivatives are exactly zero. Sampling selects an execution path; it does not authorize deleting small nonzero values. Preserve nonfinite-coefficient propagation and resource/layout fallbacks. Review `eval_rho2`, `VXC_dcontract_rho_grad_tau`, `VXCeval_rho_mgga_mo`, `test_numint_rho` and `test_numint_rho_stream`.

6. **Sparse AO-pair tasks.** The computed matrix is `M_mu,nu=sum_g bra_mu(g) w(g) ket_nu(g)`. Target32-AO output tiles improve granularity, but complete shells can be wider than32. Scratch must use actual maximum partition width. Each AO pair has one owner, and its inner grid accumulation order is retained. Check weighted/unweighted/tau variants, nonsymmetric matrices, signed weights, partial blocks, wide shells, accumulation into existing output, and one/four-thread comparisons in `test_numint_pair_tiles`.

7. **Fused potential assembly.** With weights already including quadrature and XC derivatives:

   ```text
   V_mu,nu = sum_g [w_rho phi_mu phi_nu
                  + w_grad dot ((grad phi_mu) phi_nu + phi_mu grad phi_nu)
                  + (w_tau/2) grad phi_mu dot grad phi_nu]
   ```

   LDA/GGA/MGGA use one/two/four packed channels and64-AO C tiles. The helper accepts raw weights; callers must not retain old half-factors intended for a later Hermitian merge. Existing output must neither be overwritten nor accidentally symmetrized again. Signed dense tau weights can be represented as positive and negative Gram matrices; different bra/ket cases require the proper cross contraction. Resource checks must decline before modifying output. The total fused workspace cap is128MiB, which is not a bound on whole-process RSS. Actual-width scratch from step6 is a prerequisite. Review `_xc_vmat`, `_mgga_vmat`, `mgga_vmat.c`, the tau kernels and all three potential/tau/response-assembly test modules.

8. **Blocked public cache construction.** The lifetime is AO blocks -> final density -> release AO buffers -> derivative blocks -> full public arrays. Default public calls return full arrays; compact storage is an internal opt-in, and the cache has no universal128MiB limit. Preserve callback invocation semantics, array dtype promotion, single-block lifetime, spin containers and Laplacian handling. Custom or overridden XC callbacks retain whole-grid behavior. Review `_cache_xc_kernel_blocks`, both public cache entry points and `test_numint_cache_blocks`.

9. **Compact internal response.** Let z be Cartesian density variables, u(z) LibXC invariants, and e(u) the energy per volume. The exact linear action is

   ```text
   delta v_z = J^T H_e (J delta z)
               + sum_a (de/du_a) H_(u_a) delta z,    J = du/dz.
   ```

   The second geometric term must remain. For RKS, `sigma=grad rho dot grad rho` has Hessian2I. UKS has aa/ab/bb invariants and cross-spin terms. Packed off-diagonal Hessian entries contribute at both symmetric positions. Singlet/triplet projection and spin scaling retain the established factors. Arrays captured by multiple closures must not be mutated. Compact storage is an exact representation change, not a low-rank or cross-SCF approximation. Complex/Laplacian/custom-backend cases retain their public-array fallbacks. Supplied `dm0`, explicit grids and secondary grids keep their API semantics. Retain step7's AO assembly when adding compact weight application. Review `xc_cache.py`, `_assemble_fxc`, `_gen_response` and `test_xc_cache`.

The stage-specific tests validate changed features; the final109-test suite checks their combination. Fresh SCF benchmarking measures each contribution after its cumulative parent, so the increments depend on this order. It does not establish isolated global attribution or guarantee gains for all molecules, functionals and thread counts. The response stages also need actual cache/CPKS time, sampled RSS, cache payload bytes, residual and polarizability comparisons; ground-state SCF alone cannot establish their intended memory benefit.

No numerical threshold, grid, FP64 setting or default direct J/K method is relaxed. No fast-math is enabled. All source identities and native builds are bound to the measured commits. A rebase that changes code needs renewed validation. The human author must be able to explain the equations, factors, output slots, eligibility checks and thread ownership in the corresponding C/Python implementations.

[AI_POLICY.md](https://github.com/pyscf/pyscf/blob/master/AI_POLICY.md) requires substantial AI assistance to be disclosed and generated contributions to be reviewed and submitted by a human. OpenAI Codex substantially assisted implementation, tests, automation, code review and documentation.
