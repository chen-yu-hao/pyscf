# Analytic CPKS benchmark

Accepted: **True**; results **18/18**.

| Case | Threads | Version | CPKS total s | Cache s | Solver s | Peak RSS MiB | RSS increment MiB | vs base | vs parent |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| water4-rks | 8 | baseline | 12.6489 | 0.2162 | 12.4377 | 610.4 | 506.2 | 1.000x | 1.000x |
| water4-rks | 8 | cacheblocks | 12.5529 | 0.2335 | 12.3320 | 605.5 | 500.8 | 1.008x | 1.008x |
| water4-rks | 8 | compact | 12.6182 | 0.1666 | 12.4534 | 584.3 | 480.0 | 1.002x | 0.995x |
| oh-water3-uks | 8 | baseline | 27.2893 | 0.4923 | 26.7768 | 660.0 | 556.2 | 1.000x | 1.000x |
| oh-water3-uks | 8 | cacheblocks | 27.2220 | 0.5388 | 26.6832 | 674.9 | 571.0 | 1.002x | 1.002x |
| oh-water3-uks | 8 | compact | 27.2283 | 0.2826 | 26.9592 | 604.1 | 500.1 | 1.002x | 1.000x |

Medians of three fresh processes when the campaign uses its default three repeats. Minimum and maximum timings, commit identifiers, persistent cache bytes, and all numerical gates are in acceptance.json. Full cache construction and iterative Coulomb/XC response are included. SCF, imports and input loading are excluded from timing.

VmRSS sampled every 5 ms can miss brief transients; process HWM includes imports and input loading

Shared host. Busy CPU seconds on selected cores and SMT siblings minus child CPU seconds are a diagnostic, not proof of exclusive occupancy.
