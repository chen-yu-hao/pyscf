#!/usr/bin/env bash
# Build a matched candidate/upstream pair without sharing source roots between profiles.
set -euo pipefail

if [[ $# != 1 || ( $1 != portable && $1 != native ) ]]; then
    printf 'Usage: %s portable|native\n' "$0" >&2
    exit 2
fi
profile=$1
: "${CANDIDATE_SHA:?Set the full candidate commit SHA}"
: "${UPSTREAM_SHA:?Set the full official upstream commit SHA}"
: "${GITHUB_WORKSPACE:?Set the Actions checkout directory}"
: "${RUNNER_TEMP:?Set the runner temporary directory}"
: "${GITHUB_ENV:?Set the Actions environment output file}"
for revision in "$CANDIDATE_SHA" "$UPSTREAM_SHA"; do
    if [[ ! $revision =~ ^[0-9a-fA-F]{40}$ ]]; then
        printf 'Expected a full 40-character commit SHA: %s\n' "$revision" >&2
        exit 2
    fi
done
CANDIDATE_SHA=${CANDIDATE_SHA,,}
UPSTREAM_SHA=${UPSTREAM_SHA,,}
build_jobs=${COMBO_BUILD_JOBS:-3}
if [[ ! $build_jobs =~ ^[1-9][0-9]*$ ]]; then
    printf 'COMBO_BUILD_JOBS must be a positive integer.\n' >&2
    exit 2
fi
if [[ $(uname -s) != Linux ]]; then
    printf 'This benchmark build/provenance script requires a Linux runner.\n' >&2
    exit 2
fi

workspace=$(realpath "$GITHUB_WORKSPACE")
runner_temp=$(realpath "$RUNNER_TEMP")
upstream_checkout="$workspace/upstream"
candidate_root="$runner_temp/pyscf-combo-candidate-$profile"
upstream_root="$runner_temp/pyscf-combo-upstream-$profile"
results="$workspace/combo-results"
evidence="$results/build-$profile"
roots_file="$results/roots-$profile.json"
for value in "$workspace" "$runner_temp" "$GITHUB_ENV"; do
    if [[ $value == *$'\n'* || $value == *$'\r'* ]]; then
        printf 'Runner paths must not contain line breaks.\n' >&2
        exit 2
    fi
done

[[ $(git -C "$workspace" rev-parse --show-toplevel) == "$workspace" ]]
[[ $(git -C "$upstream_checkout" rev-parse --show-toplevel) == "$upstream_checkout" ]]
[[ $(git -C "$workspace" rev-parse "$CANDIDATE_SHA^{commit}") == "$CANDIDATE_SHA" ]]
[[ $(git -C "$upstream_checkout" rev-parse "$UPSTREAM_SHA^{commit}") == "$UPSTREAM_SHA" ]]
for path in "$candidate_root" "$upstream_root" "$evidence" "$roots_file"; do
    if [[ -e $path || -L $path ]]; then
        printf 'Refusing to reuse an existing build/profile path: %s\n' "$path" >&2
        exit 2
    fi
done
mkdir -p "$evidence"

# CMake emits shared objects into each source tree, so separate build directories
# alone would not isolate portable/native binaries. Detached worktrees do.
git -C "$workspace" worktree add --detach "$candidate_root" "$CANDIDATE_SHA"
git -C "$upstream_checkout" worktree add --detach "$upstream_root" "$UPSTREAM_SHA"
[[ $(git -C "$candidate_root" rev-parse HEAD) == "$CANDIDATE_SHA" ]]
[[ $(git -C "$upstream_root" rev-parse HEAD) == "$UPSTREAM_SHA" ]]

c_compiler=$(command -v "${CC:-cc}")
strict_flags='-O2 -DNDEBUG -fno-fast-math -ffp-contract=off'
native_flags='-march=native -funroll-loops -ftree-vectorize -fno-math-errno'
march_native=OFF
upstream_core_flags=''
if [[ $profile == native ]]; then
    march_native=ON
    # Explicit core flags also cover upstream revisions that do not implement
    # BUILD_MARCH_NATIVE. If supported, repeated identical options are harmless.
    upstream_core_flags=$native_flags
fi
# Do not silently inherit extra optimization/link flags from the runner shell.
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS

"$c_compiler" --version > "$evidence/compiler.txt"
cmake --version > "$evidence/cmake-version.txt"
python -VV > "$evidence/python-version.txt" 2>&1
python -m pip freeze > "$evidence/python-packages.txt"
uname -a > "$evidence/uname.txt"
lscpu > "$evidence/cpu.txt"
if command -v numactl >/dev/null 2>&1; then
    numactl --hardware > "$evidence/numa.txt"
fi
python - "$evidence" <<'PY'
import json
import os
from pathlib import Path
import sys

out = Path(sys.argv[1])
data = {"affinity": sorted(os.sched_getaffinity(0)), "os_cpu_count": os.cpu_count()}
for filename in ("/proc/self/cgroup", "/sys/fs/cgroup/cpu.max",
                 "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
                 "/sys/fs/cgroup/cpu/cpu.cfs_period_us"):
    path = Path(filename)
    if path.is_file():
        data[filename] = path.read_text().strip()
(out / "cpu-allocation.json").write_text(json.dumps(data, indent=2) + "\n")
PY

common_cmake=(
    -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_C_COMPILER=$c_compiler"
    "-DCMAKE_C_FLAGS_RELEASE=$strict_flags"
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    "-DBUILD_MARCH_NATIVE=$march_native"
    -DUSE_QCINT=OFF
    -DBLA_VENDOR=OpenBLAS
    -DENABLE_OPENMP=ON
    -DENABLE_XCFUN=OFF
    -DBUILD_XCFUN=OFF
)
cmake -S "$candidate_root/pyscf/lib" -B "$candidate_root/build-combo" \
    "${common_cmake[@]}" -DCMAKE_C_FLAGS= \
    2>&1 | tee "$evidence/configure-candidate.log"
cmake --build "$candidate_root/build-combo" --parallel "$build_jobs" \
    2>&1 | tee "$evidence/build-candidate.log"
cp "$candidate_root/build-combo/CMakeCache.txt" "$evidence/cmake-candidate.txt"
cp "$candidate_root/build-combo/compile_commands.json" "$evidence/compile-commands-candidate.json"
cp "$candidate_root/build-combo/deps/src/libcint-build/CMakeCache.txt" "$evidence/cmake-libcint.txt"
cp "$candidate_root/build-combo/deps/src/libxc-build/CMakeCache.txt" "$evidence/cmake-libxc.txt"
git -C "$candidate_root/build-combo/deps/src/libcint" rev-parse HEAD \
    > "$evidence/libcint-source.txt"

[[ -d $candidate_root/pyscf/lib/deps ]]
[[ ! -e $upstream_root/pyscf/lib/deps && ! -L $upstream_root/pyscf/lib/deps ]]
ln -s "$candidate_root/pyscf/lib/deps" "$upstream_root/pyscf/lib/deps"
cmake -S "$upstream_root/pyscf/lib" -B "$upstream_root/build-combo" \
    "${common_cmake[@]}" "-DCMAKE_C_FLAGS=$upstream_core_flags" \
    -DBUILD_LIBCINT=OFF -DBUILD_LIBXC=OFF \
    2>&1 | tee "$evidence/configure-upstream.log"
cmake --build "$upstream_root/build-combo" --parallel "$build_jobs" \
    2>&1 | tee "$evidence/build-upstream.log"
cp "$upstream_root/build-combo/CMakeCache.txt" "$evidence/cmake-upstream.txt"
cp "$upstream_root/build-combo/compile_commands.json" "$evidence/compile-commands-upstream.json"

for role in candidate upstream; do
    root=$candidate_root
    if [[ $role == upstream ]]; then root=$upstream_root; fi
    for name in libcgto.so libcvhf.so libdft.so libxc_itrf.so; do
        [[ -f $root/pyscf/lib/$name ]]
    done
    ldd "$root/pyscf/lib/libcgto.so" "$root/pyscf/lib/libcvhf.so" \
        "$root/pyscf/lib/libdft.so" "$root/pyscf/lib/libxc_itrf.so" \
        > "$evidence/ldd-$role.txt"
    git -C "$root" status --porcelain --untracked-files=no > "$evidence/tracked-status-$role.txt"
    git -C "$root" diff --quiet HEAD --
done

# Check compiler commands and the libraries actually loaded in separate Python
# processes. Only publish roots/GITHUB_ENV after every build identity check passes.
python - "$profile" "$candidate_root" "$upstream_root" "$CANDIDATE_SHA" \
    "$UPSTREAM_SHA" "$evidence" "$roots_file" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

profile, candidate, upstream, candidate_sha, upstream_sha, evidence, roots_file = sys.argv[1:]
evidence = Path(evidence)
deps = (Path(candidate) / "pyscf/lib/deps").resolve()
require_native = {"-march=native", "-funroll-loops", "-ftree-vectorize", "-fno-math-errno"}
required = {"-O2", "-DNDEBUG", "-fno-fast-math", "-ffp-contract=off"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_cache(path):
    values = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.split(":", 1)[0]] = value
    return values


probe = r'''
import hashlib, json, os
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import pyscf
from pyscf import dft, gto, lib, scf
from threadpoolctl import threadpool_info
if not Path(pyscf.__file__).resolve().is_relative_to(root):
    raise RuntimeError("Wrong PySCF import: " + pyscf.__file__)
lib.num_threads(2)
files = set()
for line in Path("/proc/self/maps").read_text().splitlines():
    name = line.split()[-1]
    path = Path(name)
    if name.startswith("/") and path.is_file() and (
        path.name.startswith(("libcint.", "libxc.", "libxc_itrf.", "libcvhf.", "libdft.", "libcgto."))
        or "openblas" in path.name or path.name.startswith("libgomp.")
    ):
        files.add(path.resolve())
libraries = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(files)]
pools = threadpool_info()
for pool in pools:
    path = Path(pool["filepath"])
    pool["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
print(json.dumps({"source_file": pyscf.__file__, "libraries": libraries,
                  "threadpools": pools, "openmp_probe_threads": lib.num_threads()}))
'''

manifest = {"build_profile": profile, "evidence_dir": str(evidence), "build_validation": {}}
backend_reference = None
blas_reference = None
for role, root_text, sha in (("candidate", candidate, candidate_sha),
                             ("upstream", upstream, upstream_sha)):
    root = Path(root_text).resolve()
    actual_sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    require(actual_sha == sha, f"{role}: source SHA changed")
    require((root / "pyscf/lib/deps").resolve() == deps, f"{role}: dependencies are not shared")
    cache = read_cache(evidence / f"cmake-{role}.txt")
    require(cache.get("CMAKE_BUILD_TYPE", "").lower() == "release", f"{role}: not a Release build")
    require(cache.get("BLA_VENDOR") == "OpenBLAS", f"{role}: unexpected BLAS vendor")
    require(cache.get("USE_QCINT") == "OFF", f"{role}: QCINT unexpectedly enabled")
    require(cache.get("ENABLE_XCFUN") == "OFF", f"{role}: XCFun unexpectedly enabled")
    require(cache.get("BUILD_MARCH_NATIVE") == ("ON" if profile == "native" else "OFF"),
            f"{role}: native build option differs from requested profile")
    if role == "upstream":
        require(cache.get("BUILD_LIBCINT") == "OFF" and cache.get("BUILD_LIBXC") == "OFF",
                "Upstream must reuse candidate dependencies")
    commands = json.loads((evidence / f"compile-commands-{role}.json").read_text())
    checked, distinct_flags, openmp_commands = 0, set(), 0
    for entry in commands:
        if Path(entry["file"]).suffix != ".c":
            continue
        tokens = entry.get("arguments") or shlex.split(entry["command"])
        require(required.issubset(tokens), f"{role}: missing strict flags in {entry['file']}")
        optimizations = [x for x in tokens if x in ("-O0", "-O1", "-O2", "-O3", "-Og", "-Os", "-Oz", "-Ofast")]
        require(optimizations[-1:] == ["-O2"], f"{role}: -O2 was overridden in {entry['file']}")
        require("-ffast-math" not in tokens and "-Ofast" not in tokens,
                f"{role}: unsafe fast-math in {entry['file']}")
        contractions = [x for x in tokens if x.startswith("-ffp-contract=")]
        require(contractions[-1:] == ["-ffp-contract=off"], f"{role}: FP contraction changed")
        if profile == "native":
            require(require_native.issubset(tokens), f"{role}: missing actual native flags in {entry['file']}")
        else:
            require("-march=native" not in tokens, f"{role}: native flag in portable build")
        distinct_flags.add(tuple(x for x in tokens if x.startswith(("-O", "-f", "-m", "-DNDEBUG"))))
        openmp_commands += int("-fopenmp" in tokens)
        checked += 1
    require(checked > 0 and openmp_commands > 0, f"{role}: no C/OpenMP compiler evidence")
    require("not found" not in (evidence / f"ldd-{role}.txt").read_text(), f"{role}: unresolved shared library")
    flags = {"checked_c_commands": checked, "openmp_commands": openmp_commands,
             "distinct_flags": [list(x) for x in sorted(distinct_flags)]}
    (evidence / f"verified-flags-{role}.json").write_text(json.dumps(flags, indent=2) + "\n")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               BLIS_NUM_THREADS="1", OMP_DYNAMIC="FALSE", OMP_MAX_ACTIVE_LEVELS="1")
    loaded = json.loads(subprocess.check_output([sys.executable, "-I", "-c", probe, str(root)],
                                               env=env, text=True))
    (evidence / f"loaded-libraries-{role}.json").write_text(json.dumps(loaded, indent=2) + "\n")
    require(loaded["openmp_probe_threads"] == 2, f"{role}: OpenMP thread control unavailable")
    for prefix in ("libcgto.", "libcvhf.", "libdft.", "libxc_itrf."):
        entries = [x for x in loaded["libraries"] if Path(x["path"]).name.startswith(prefix)]
        require(bool(entries), f"{role}: core library {prefix} was not loaded")
        require(all(Path(x["path"]).is_relative_to(root / "pyscf/lib") for x in entries),
                f"{role}: core library {prefix} resolved outside its own source root")
    backend = {}
    for prefix in ("libcint.", "libxc."):
        entries = [x for x in loaded["libraries"] if Path(x["path"]).name.startswith(prefix)]
        require(bool(entries), f"{role}: {prefix} was not loaded")
        require(all(Path(x["path"]).is_relative_to(deps) for x in entries),
                f"{role}: {prefix} resolved outside shared profile dependencies")
        backend[prefix] = sorted({x["sha256"] for x in entries})
    blas = sorted({x["sha256"] for x in loaded["threadpools"] if x["user_api"] == "blas"})
    require(bool(blas), f"{role}: no loaded BLAS evidence")
    require(all(x["num_threads"] == 1 for x in loaded["threadpools"] if x["user_api"] == "blas"),
            f"{role}: BLAS thread cap not honored in build probe")
    if backend_reference is None:
        backend_reference, blas_reference = backend, blas
    else:
        require(backend == backend_reference, "Candidate/upstream backend hashes differ")
        require(blas == blas_reference, "Candidate/upstream loaded BLAS hashes differ")
    manifest[role] = {"root": str(root), "sha": sha, "build_profile": profile}
    manifest["build_validation"][role] = flags

require(Path(candidate).resolve() != Path(upstream).resolve(), "Source roots overlap")
manifest.update(backend_hashes=backend_reference, blas_hashes=blas_reference,
                shared_deps=str(deps), build_verified=True,
                numerical_validation="Required separately; building/importing is not a numerical test")
(evidence / "build-verification.json").write_text(json.dumps(manifest, indent=2) + "\n")
destination = Path(roots_file)
temporary = destination.with_suffix(".json.tmp")
temporary.write_text(json.dumps(manifest, indent=2) + "\n")
temporary.replace(destination)
print(json.dumps({"build_profile": profile, "build_verified": True,
                  "candidate": manifest["candidate"], "upstream": manifest["upstream"],
                  "backend_hashes": backend_reference, "roots_file": str(destination)}, indent=2))
PY

printf '%s=%s\n' "CANDIDATE_ROOT_${profile^^}" "$candidate_root" >> "$GITHUB_ENV"
printf '%s=%s\n' "UPSTREAM_ROOT_${profile^^}" "$upstream_root" >> "$GITHUB_ENV"
