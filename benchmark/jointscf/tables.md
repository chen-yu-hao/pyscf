## Environment

| | |
|---|---|
| cpu | Intel(R) Xeon(R) Processor @ 2.10GHz |
| ncpu | 4 |
| mem_gb | 15.7 |
| platform | Linux-6.18.44-fc-v470-x86_64-with-glibc2.39 |
| python | 3.13.18 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| pyscf | 2.14.0 |
| git | 30f225486771561b7e4512516c06e3e7b965ac24 |

## Table 1. Fock builds per configuration and accuracy

All methods use the same convergence criteria (conv_tol = 1e-9 Ha, gradient 3.2e-5).  E_ref: independent calculations converged to 1e-12 Ha.

| ensemble | level of theory | N_AO | direct | JointSCF sync | JointSCF cascade | direct / cascade | converged | max\|E_direct − E_ref\| | max\|E_cascade − E_ref\| |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H2O ×16, random | PBE/def2-SVP | 24 | 9.88 | 9.88 | 8.38 | 1.18× | 16/16 | 2.6e-11 | 2.6e-11 |
| H2O ×32, random | PBE/def2-SVP | 24 | 9.97 | 9.97 | 7.25 | 1.38× | 32/32 | 2.6e-11 | 1.5e-11 |
| H2O ×64, random | PBE/def2-SVP | 24 | 9.97 | 9.97 | 6.22 | 1.60× | 64/64 | 2.6e-11 | 1.1e-10 |
| H2O ×128, random | PBE/def2-SVP | 24 | 9.99 | 9.99 | 5.23 | 1.91× | 128/128 | 2.6e-11 | 1.3e-10 |
| H2O ×64, 2D scan | PBE/def2-SVP | 24 | 9.44 | 9.44 | 4.97 | 1.90× | 64/64 | 1.1e-11 | 7.7e-11 |
| H2O ×64, stretched | PBE/def2-SVP | 24 | 11.08 | 11.08 | 7.67 | 1.44× | 64/64 | 8.5e-10 | 1.1e-09 |
| C2H4 ×48, path | PBE/def2-SVP | 48 | 10.00 | 10.00 | 5.96 | 1.68× | 48/48 | 1.0e-10 | 1.0e-10 |
| C2H4 ×48, random | PBE/def2-SVP | 48 | 10.02 | 10.02 | 9.08 | 1.10× | 48/48 | 1.2e-10 | 3.9e-10 |
| H2CO ×40, random | B3LYP/def2-TZVP | 74 | 10.97 | 10.97 | 9.22 | 1.19× | 40/40 | 4.4e-10 | 3.9e-11 |

## Table 2. Wall time on one core (seconds; median and range of the repeats)

| ensemble | direct | JointSCF sync | JointSCF cascade | speed-up | direct + AO cache | cascade + AO cache | speed-up (both with cache) | cascade + cache vs direct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H2O ×16, random | 6.44 (6.41–6.51) | 6.36 | 5.44 (5.38–5.61) | 1.18× | 4.85 (4.57–4.99) | 4.29 (4.28–4.44) | 1.13× | 1.50× |
| H2O ×32, random | 13.02 (12.57–13.08) | 12.63 | 8.96 (8.75–9.69) | 1.45× | 9.66 (9.66–9.91) | 7.52 (7.45–7.56) | 1.28× | 1.73× |
| H2O ×64, random | 26.45 (26.21–26.72) | 26.65 | 17.00 (16.85–17.19) | 1.56× | 19.85 (19.56–19.89) | 13.85 (13.46–14.12) | 1.43× | 1.91× |
| H2O ×128, random | 51.47 (49.81–52.05) | 53.28 | 28.36 (28.23–28.59) | 1.82× | 38.68 (37.93–39.62) | 23.94 (23.60–24.59) | 1.62× | 2.15× |
| H2O ×64, 2D scan | 24.93 (24.29–25.70) | 24.44 | 14.14 (13.75–14.17) | 1.76× | 18.89 (18.77–18.96) | 11.50 (11.47–12.07) | 1.64× | 2.17× |
| H2O ×64, stretched | 28.71 (28.56–29.14) | 28.38 | 20.11 (19.94–20.70) | 1.43× | 21.20 (21.17–21.83) | 15.92 (15.83–16.76) | 1.33× | 1.80× |
| C2H4 ×48, path | 47.58 (47.25–49.43) | 44.91 | 29.04 (28.92–30.38) | 1.64× | 32.42 (31.43–32.89) | 22.76 (21.39–23.04) | 1.42× | 2.09× |
| C2H4 ×48, random | 47.87 (46.70–49.58) | 47.36 | 44.46 (42.18–44.55) | 1.08× | 33.99 (32.93–34.51) | 31.41 (30.69–31.46) | 1.08× | 1.52× |
| H2CO ×40, random | 58.05 (57.96–60.42) | 63.33 | 52.05 (51.33–52.20) | 1.12× | 41.83 (41.36–42.42) | 57.82 (56.42–57.94) | 0.72× | 1.00× |

## Table 3. Wall time on 4 cores (seconds; median and range of the repeats)

direct (4 threads): one calculation at a time with 4 OpenMP threads (the default behaviour of PySCF).  direct (4 processes): 4 independent single-thread calculations at a time.  The start-up of the worker processes is not included (Table 4).

| ensemble | direct (4 threads) | direct (4 processes) | JointSCF sync | JointSCF cascade | speed-up vs 4 processes | direct (4 proc.) + cache | cascade + cache | speed-up (both with cache) | cascade + cache vs direct 1 core |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H2O ×16, random | 18.03 | 1.63 (1.63–1.67) | 1.82 (1.80–1.83) | 1.64 (1.57–1.65) | 1.00× | 1.30 (1.28–1.35) | 1.29 (1.26–1.40) | 1.00× | 5.0× |
| H2O ×32, random | 38.37 | 3.23 (3.17–3.26) | 3.33 (3.33–3.43) | 2.73 (2.67–2.74) | 1.18× | 2.48 (2.47–2.55) | 2.32 (2.17–2.36) | 1.07× | 5.6× |
| H2O ×64, random | 79.54 | 6.78 (6.60–6.83) | 6.85 (6.73–6.92) | 5.21 (5.06–5.39) | 1.30× | 5.07 (5.03–5.20) | 4.37 (3.94–4.42) | 1.16× | 6.1× |
| H2O ×128, random | 157.27 | 12.99 (12.83–13.85) | 13.22 (13.20–13.23) | 9.02 (8.80–9.53) | 1.44× | 9.74 (9.62–10.18) | 8.34 (7.88–8.36) | 1.17× | 6.2× |
| H2O ×64, 2D scan | 72.56 | 6.44 (6.44–6.55) | 6.53 (6.53–6.58) | 4.21 (3.96–4.33) | 1.53× | 4.90 (4.88–4.98) | 3.65 (3.40–3.94) | 1.34× | 6.8× |
| H2O ×64, stretched | 79.12 | 7.23 (7.15–7.47) | 7.40 (7.31–7.48) | 5.82 (5.65–5.98) | 1.24× | 5.56 (5.46–5.61) | 4.68 (4.63–4.94) | 1.19× | 6.1× |
| C2H4 ×48, path | 62.73 | 11.86 (11.58–12.48) | 11.86 (11.80–12.29) | 9.57 (9.48–9.72) | 1.24× | 8.52 (8.19–8.57) | 7.03 (6.85–7.89) | 1.21× | 6.8× |
| C2H4 ×48, random | 63.61 | 12.52 (12.14–12.97) | 12.81 (12.50–13.16) | 13.76 (13.03–14.49) | 0.91× | 8.69 (8.33–8.83) | 9.45 (9.10–9.94) | 0.92× | 5.1× |
| H2CO ×40, random | 74.91 | 15.05 (14.87–16.22) | 15.47 (15.22–16.33) | 16.93 (16.21–17.34) | 0.89× | 10.90 (10.83–11.00) | 12.70 (12.07–12.79) | 0.86× | 4.6× |

## Table 4. Start-up of the worker processes and peak memory

| ensemble | start-up direct (4 proc.) / s | start-up JointSCF (4 workers) / s | peak RSS direct 1 core / MB | peak RSS cascade 1 worker / MB | peak RSS cascade 1 worker + cache / MB |
|---|---:|---:|---:|---:|---:|
| H2O ×16, random | 0.52 | 0.39 | 154 | 173 | 580 |
| H2O ×32, random | 0.50 | 0.40 | 165 | 203 | 999 |
| H2O ×64, random | 0.52 | 0.44 | 168 | 265 | 1847 |
| H2O ×128, random | 0.50 | 0.48 | 174 | 394 | 3553 |
| H2O ×64, 2D scan | 0.51 | 0.44 | 167 | 267 | 1847 |
| H2O ×64, stretched | 0.52 | 0.45 | 170 | 266 | 1847 |
| C2H4 ×48, path | 0.55 | 0.43 | 295 | 568 | 3650 |
| C2H4 ×48, random | 0.53 | 0.46 | 297 | 574 | 3645 |
| H2CO ×40, random | 0.55 | 0.51 | 643 | 1515 | 4197 |

## Table 5. Cascade schedule

Levels: number of configurations / SCF cycle of the promotion / mean number of Fock builds.  Model error: leave-one-out error of the cross-configuration model at the promotions (norm of the commutator [F, D]).

| ensemble | descriptors | levels (size / cycle / Fock builds) | model error at the promotions |
|---|---:|---|---|
| H2O ×16, random | 3 | 8 / 0 / 9.75, 8 / 4 / 7.00 | 5.0e-02 (8 src) |
| H2O ×32, random | 3 | 8 / 0 / 10.00, 8 / 4 / 7.12, 16 / 6 / 5.94 | 5.0e-02 (8 src), 1.0e-02 (16 src) |
| H2O ×64, random | 3 | 8 / 0 / 10.12, 8 / 4 / 7.25, 16 / 6 / 6.06, 32 / 8 / 5.06 | 8.8e-02 (8 src), 1.7e-02 (16 src), 1.4e-03 (32 src) |
| H2O ×128, random | 3 | 8 / 0 / 10.00, 8 / 4 / 7.25, 16 / 6 / 6.00, 32 / 8 / 5.12, 64 / 10 / 4.25 | 1.3e-01 (8 src), 1.6e-02 (16 src), 2.1e-03 (32 src), 2.3e-04 (64 src) |
| H2O ×64, 2D scan | 2 | 8 / 0 / 9.50, 8 / 5 / 5.62, 16 / 7 / 4.25, 32 / 9 / 4.03 | 1.2e-02 (8 src), 5.7e-04 (16 src), 2.9e-05 (32 src) |
| H2O ×64, stretched | 3 | 8 / 0 / 11.00, 8 / 4 / 8.12, 16 / 6 / 7.69, 32 / 9 / 6.72 | 3.6e-01 (8 src), 1.1e-01 (16 src), 2.2e-02 (32 src) |
| C2H4 ×48, path | 10 | 12 / 0 / 10.00, 12 / 5 / 5.83, 24 / 8 / 4.00 | 1.0e-02 (12 src), 1.4e-04 (24 src) |
| C2H4 ×48, random | 11 | 13 / 0 / 10.00, 13 / 4 / 9.08, 22 / 8 / 8.55 | 1.5e-01 (13 src), 5.3e-02 (26 src) |
| H2CO ×40, random | 6 | 8 / 0 / 11.00, 8 / 3 / 9.00, 16 / 5 / 8.69, 8 / 8 / 8.75 | 1.3e-01 (8 src), 5.9e-02 (16 src), 3.0e-02 (32 src) |
