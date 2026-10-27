"""Single fresh-process SCF; parent measures identical process boundaries."""
import argparse,json,os,resource,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--xc',default='MN15');p.add_argument('--cores',type=int,default=8);p.add_argument('--cutoff',type=float,default=1e-10);args=p.parse_args()
import numpy as np
import pyscf
from pyscf import gto,dft,lib
from threadpoolctl import threadpool_info
lib.num_threads(args.cores)
spec=next(c for c in json.load(open(Path(__file__).with_name('protocol.json')))['cases'] if c['id']==args.case)
mol=gto.M(atom=spec['atoms_bohr'],unit='Bohr',basis=spec['basis'],spin=spec['spin'],charge=spec['charge'],cart=False,symmetry=False,max_memory=2000,verbose=0)
mf=dft.UKS(mol) if mol.spin else dft.RKS(mol)
mf.xc=args.xc;mf.grids.atom_grid=(99,590);mf.grids.prune=None;mf.small_rho_cutoff=0
mf.grids.cutoff=args.cutoff;mf._numint.cutoff=args.cutoff*100
mf.conv_tol=1e-10;mf.conv_tol_grad=3e-5;mf.max_cycle=80;mf.diis_space=12
mf.init_guess='huckel';mf.direct_scf=True;mf._is_mem_enough=lambda:False;mf.chkfile=None
history=[]
def strict(env):
 dd=np.asarray(env['dm'])-np.asarray(env['dm_last'])
 row=dict(cycle=int(env['cycle']+1),density_rms=float(np.sqrt(np.mean(dd*dd))),density_max=float(np.max(abs(dd))),energy_delta=float(abs(env['e_tot']-env['last_hf_e'])),gradient=float(env['norm_gorb']))
 history.append(row)
 return row['density_rms']<1e-9 and row['density_max']<1e-7 and row['energy_delta']<1e-7 and row['gradient']<3e-5
mf.check_convergence=strict
start=time.perf_counter();energy=mf.kernel();kernel=time.perf_counter()-start;cycles=mf.cycles;polish=0
if not mf.converged:
 t=time.perf_counter();energy=mf.kernel(dm0=mf.make_rdm1());kernel+=time.perf_counter()-t;polish=mf.cycles;cycles+=polish
dm=np.asarray(mf.make_rdm1());overlap=mf.get_ovlp();ne=np.einsum('...ij,ji->...',dm,overlap)
record={'case':args.case,'basis':spec['basis'],'xc':args.xc,'spin':mol.spin,'cores':args.cores,'python':__import__('sys').executable,'pyscf_path':pyscf.__file__,'version':pyscf.__version__,'affinity':sorted(os.sched_getaffinity(0)),'energy':float(energy),'kernel_wall_s':kernel,'cycles':int(cycles),'polish_cycles':int(polish),'strict':bool(mf.converged),'history':history,'grid_nominal':[99,590],'grid_points':len(mf.grids.weights),'nonzero_weight_points':int(np.count_nonzero(mf.grids.weights)),'cutoff':args.cutoff,'numint_cutoff':args.cutoff*100,'nelec_overlap':np.asarray(ne).tolist(),'nao':mol.nao,'cpu_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'threadpools':threadpool_info(),'native_maps':sorted(set(x.split()[-1] for x in Path('/proc/self/maps').read_text().splitlines() if '.so' in x and '/pyscf/' in x))}
if mol.spin:record['spin_square']=list(map(float,mf.spin_square()))
args.output.parent.mkdir(parents=True,exist_ok=True)
np.save(args.output.with_suffix('.dm.npy'),dm)
# One geometry/grid identity record; no large full-grid arrays in timed output.
import hashlib
record['grid_coords_digest']=hashlib.sha256(np.ascontiguousarray(mf.grids.coords).tobytes()).hexdigest();record['grid_weights_digest']=hashlib.sha256(np.ascontiguousarray(mf.grids.weights).tobytes()).hexdigest()
args.output.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['case','energy','kernel_wall_s','strict','cycles']}),flush=True)
if not record['strict']:raise SystemExit(2)
