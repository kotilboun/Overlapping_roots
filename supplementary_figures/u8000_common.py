from __future__ import annotations
import importlib.util, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import numpy as np

MODULE_PATH = Path(__file__).resolve().parent / 'generate_abm_mean_trajectory_error_convergence.py'
spec = importlib.util.spec_from_file_location('abm_validation_u8000_core', MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

TAU=np.linspace(0.0,5.0,151)
SWITCH_TAU=0.5
U0=8000
I0=160
REPS=120
HIGH_K=40
CHECK_K=80
BASE_SEED=20260731
REPLICATE_SIZES=(1,2,5,10,20,40,80,120)
SUBSET_DRAWS=30

@dataclass(frozen=True)
class ProtocolSpec:
    protocol_id: str
    label: str
    Gamma: float
    d: float
    beta: float
    gamma: float
    gamma_c: float
    p_f: float
    R: float
    c: float
    U0: int
    I0: int

def make_protocol(protocol_id,label,R,c,Gamma,d):
    pf=0.0 if c==0.0 else c/d
    return ProtocolSpec(protocol_id,label,Gamma,d,R*Gamma/U0,(1-d)*Gamma,d*Gamma,pf,R,c,U0,I0)

def build_grid_protocols(R,c):
    if c==1.0:
        return [make_protocol('A','fast complete tracing',R,c,2.0,1.0), make_protocol('B','slow complete tracing',R,c,0.5,1.0)]
    if c==0.0:
        return [make_protocol('A','fast no tracing',R,c,2.0,1.0), make_protocol('B','slow no tracing',R,c,0.5,0.25)]
    return [make_protocol('A','frequent partial tracing',R,c,2.0,1.0), make_protocol('B','less-frequent complete tracing',R,c,0.5,c)]

def canonical_protocol(R,c):
    return make_protocol('canonical','canonical decomposition',R,c,1.0,1.0)

def matched_protocols(R=4.0,c=0.5):
    return [make_protocol('A','frequent partial tracing',R,c,1.0,1.0), make_protocol('B','less-frequent complete tracing',R,c,1.0,c)]

def to_mod(p):
    return mod.RawProtocol(protocol_id=p.protocol_id,Gamma=p.Gamma,d=p.d,beta=p.beta,gamma=p.gamma,gamma_c=p.gamma_c,p_f=p.p_f,R=p.R,c=p.c,U0=p.U0,I0=p.I0)

def task(args):
    p,seed=args
    return mod.simulate_one(to_mod(p),TAU,SWITCH_TAU,seed)

def simulate_master(p,n,seed_base):
    raw=np.zeros((n,len(TAU),3),dtype=np.float32)
    tasks=[(p,seed_base+i) for i in range(n)]
    with ProcessPoolExecutor(max_workers=min(12,os.cpu_count() or 2)) as ex:
        fs={ex.submit(task,t):i for i,t in enumerate(tasks)}
        for f in as_completed(fs): raw[fs[f]]=f.result()
    return raw
