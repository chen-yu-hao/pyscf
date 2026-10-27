# Focused human review notes

This acceleration-only selection retains grid grouping, sparse AO-pair tiling, and sparse density-matrix contractions. The original MO streaming, AO evaluator, LibXC parallel region, potential assembly, and response-cache proposals remain archived separately. These notes support human review; they do not certify that the human author has reviewed or understood the code.

## Grid grouping

For clipped box coordinates i,j,k and positive widths nx,ny,nz, the key is ((i+1)*ny+(j+1))*nz+(k+1). A difference in an earlier coordinate dominates all later coordinates, so numerical key order equals the original lexicographic box order. Stable sorting keeps input order inside ties. The product range is checked with Python integers before int64 arithmetic; the original grouping implementation handles unsafe ranges. Coordinates, quadrature weights and screening are unchanged. The measured speedup belongs only to grouping.

## Sparse AO-pair tiling

The computed matrix remains M_ij=sum_g w_g a_gi b_gj. Smaller output blocks create more independently owned tasks, and dynamic scheduling uses one task at a time. Shells are not split. A shell wider than the nominal block requires scratch sized from the actual partition width; the included allocation change is necessary for the smaller nominal block. Review both weighted and unweighted paths and their ownership of output entries. No screening threshold changes.

## Sparse DM contractions

For real AOs, rho=phi^T D phi, grad(rho)=(grad phi)^T D phi+phi^T D grad(phi), and tau=(1/2)sum_a (d_a phi)^T D(d_a phi). The C implementation packs active AO/DM subsets under the existing shell/bin/pair masks and contracts the needed components together. Hermitian cases use the corresponding factor of two; unsupported nonsymmetric GGA keeps the upstream implementation. Review DM orientation, component strides and the tau factor.

The Python change limits a block's sparsity decision to that block's mask rows. Default grids, cutoffs, precision and public interfaces remain unchanged. eval_rho2 and nr_numint.c match upstream: no optimized MO path is retained in DFT-density-dm. Sparse native allocations follow existing allocation conventions; do not claim universal recovery from allocation failure.

## Evidence and submission

Performance ratios must be measured on the stated source commit and native build, using the direct upstream parent. Fixed-DM nr_rks/nr_uks timings exclude grid construction, J/K and SCF. Three process repeats describe observed variation on a shared host; they do not establish universal acceleration or statistical confidence bounds. Independent component speedups must not be multiplied together.

OpenAI Codex substantially assisted implementation, tests, benchmark automation, code review, and documentation. Under AI_POLICY.md, a human must understand and verify the code, edit the factual drafts in their own voice, and submit the upstream PR personally.
