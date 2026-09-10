"""Transcribe frozen exported formula components; do not train or evaluate models.

Generated equations use local auxiliary symbols to shorten lines. Validation
checks transcription and recombination on stored DEVELOPMENT feature rows only.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
from zipfile import ZipFile

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / 'Nuclear_Graphite_Stress_Surrogate'
FINAL = ROOT / 'outputs/18_149case_development_integration/formal_149case_integration'
BASE = ROOT / 'shared/v10_authoritative_v8_candidate5'
sources = []

def row(path):
    sources.append(path)
    with path.open() as f:
        return next(csv.DictReader(f))

names = {
    'rho': r'\rho', 'z': 'z', 'theta_cos': r'\cos\theta',
    'fluence_rate_mean': r'\overline{F}', 'fluence_rate_p95': r'F_{95}',
    'temperature_mean': r'\overline{T}', 'temperature_p95': r'T_{95}',
    'weight_loss_rate_mean': r'\overline{W}', 'weight_loss_rate_std': r's_W',
    'rho_mean': r'\overline{\rho}', 'z_mean': r'\overline{z}',
    'z_max': r'z_{\max}', 'theta_sin_std': r's_{\sin\theta}',
    'rho_fraction_proxy': r'u_\rho', 'z_fraction_proxy': r'u_z',
    'theta_fraction_proxy': r'u_\theta',
    'nearest_radial_boundary_fraction_proxy': r'd_\rho',
    'nearest_axial_boundary_fraction_proxy': r'd_z',
    'nearest_angular_boundary_fraction_proxy': r'd_\theta',
    'signed_radial_position_proxy': r'p_\rho',
    'signed_axial_position_proxy': r'p_z',
    'signed_angular_position_proxy': r'p_\theta',
    'radial_axial_signed_interaction_proxy': r'k_{\rho z}',
    'boundary_corner_proximity_proxy': 'b',
    'z_within_case_z': r'z^{*}',
}
symbols = {key: sp.Symbol(key) for key in names}
latex_names = {symbols[key]: val for key, val in names.items()}
components = []

def component(title, lhs, label, raw):
    expr = sp.sympify(raw, locals=symbols, evaluate=False)
    assert expr.free_symbols <= set(symbols.values()), expr.free_symbols - set(symbols.values())
    components.append(dict(title=title, lhs=lhs, label=label, raw=raw, expression=expr))

component('Common case-mean expression', r'\widehat{\mu}', 'common_mean',
          row(BASE/'selected_case_mean_formula.csv')['formula_original_variables'])
component('Common logarithmic stress scale', r'\widehat{\lambda}', 'common_log_scale',
          row(BASE/'selected_case_log_scale_formula.csv')['formula_original_variables'])
component('Common spatial-shape expression', r'g_{\mathrm{base}}', 'common_shape',
          row(BASE/'selected_shape_formula.csv')['formula_original_variables'])
for r in range(1,5):
    data = row(ROOT/f'outputs/10_signed_staged_residual_symbolic_pilot/iteration_{r}/selected_composite_formula.csv')
    for typ, key in [('geo','geometry'), ('phys','physical')]:
        component(f'Rotation {r}: {key} residual', rf'r_{{\mathrm{{{typ}}}}}^{{({r})}}',
                  f'{typ}_{r}', data[f'{key}_residual_formula'])
locked = row(FINAL/'locked_model_formula.csv')
component('Final case-mean correction', r'\Delta\widehat{\mu}', 'final_mean_correction',
          locked['all_development_mean_correction_formula'])
for r in range(1,4):
    data=row(ROOT/f'outputs/11_tail_aware_localised_symbolic/iteration_{r}/stages/stage_tail_localised/selected_formula.csv')
    component(f'Active local expression: rotation {r}', rf'h^{{({r})}}', f'tail_{r}',
              data['formula_original_variables'])

def latex(expr):
    result = sp.latex(expr, symbol_names=latex_names, full_prec=True, order='none')
    return result.replace(r'\left(-1\right) ', '- ')

def abbreviate(expr):
    assignments=[]
    memo={}
    def visit(node, is_root=False):
        if node.is_Atom:
            return node
        args=tuple(visit(a) for a in node.args)
        rebuilt=node.func(*args, evaluate=False)
        # Every substantial nested subexpression receives a local alias.
        if not is_root and sp.count_ops(rebuilt)>=3:
            if rebuilt not in memo:
                sym=sp.Symbol(f'aux_{len(assignments)+1}')
                latex_names[sym]=rf'a_{{{len(assignments)+1}}}'
                assignments.append((sym,rebuilt))
                memo[rebuilt]=sym
            return memo[rebuilt]
        return rebuilt
    reduced=visit(expr,True)
    reconstructed=reduced
    for sym,value in reversed(assignments):
        reconstructed=reconstructed.xreplace({sym:value})
    return assignments,reduced,reconstructed

def aligned(lhs, expr):
    terms=expr.args if expr.is_Add else (expr,)
    rendered=[]
    for index,term in enumerate(terms):
        value=latex(term)
        if index==0:
            rendered.append(lhs+' &= '+value)
        else:
            sign='- ' if value.startswith('-') else '+ '
            rendered.append(r'&\quad '+sign+(value[1:].lstrip() if value.startswith('-') else value))
    return '\n'.join(x+r' \\' for x in rendered[:-1])+'\n'+rendered[-1] if len(rendered)>1 else rendered[0]

header=HERE/'formula_appendix_intro.tex'
tex=[header.read_text()]
checks=[]
sys.path.insert(0,str(ROOT/'src'))
from signed_staged_residual_symbolic import ALL_V10_FEATURES
X=np.load(ROOT/'outputs/10_signed_staged_residual_symbolic_pilot/iteration_1/training_cache/all_v10_features.npy',mmap_mode='r')
indices=np.linspace(0,len(X)-1,4096,dtype=int)
values={sp.Symbol(name):np.asarray(X[indices,j],dtype=np.float64) for j,name in enumerate(ALL_V10_FEATURES)}

def evaluate(expr):
    keys=sorted(expr.free_symbols,key=str)
    return np.asarray(sp.lambdify(keys,expr,'numpy')(*(values[k] for k in keys)),dtype=np.float64)

for item in components:
    assignments,reduced,reconstructed=abbreviate(item['expression'])
    expected=evaluate(item['expression'])
    actual=evaluate(reconstructed)
    if expected.shape==(): expected=np.full(len(indices),float(expected))
    if actual.shape==(): actual=np.full(len(indices),float(actual))
    assert np.all(np.isfinite(expected)) and np.all(np.isfinite(actual))
    difference=float(np.max(np.abs(expected-actual)))
    assert np.allclose(expected,actual,atol=1e-10,rtol=1e-10)
    checks.append({'component':item['label'],'max_absolute_transcription_difference':difference})
    tex.append('\n'+r'\subsection*{'+item['title']+'}\n')
    if assignments:
        tex.append('The auxiliary terms below apply only to this expression.\n')
    tex.append(r'\begin{align*}'+'\n')
    equations=[aligned(latex(sym),expr) for sym,expr in assignments]
    equations.append(aligned(item['lhs'],reduced))
    tex.append((' '+r'\\'+'\n').join(equations))
    tex.append('\n'+r'\end{align*}'+'\n')
    tex.append('% Frozen source: '+item['label']+'\n')

# Check factorised primary equation against the complete frozen export.
lookup={x['label']:x['expression'] for x in components}
factorised=(lookup['common_mean']+lookup['final_mean_correction']+
            sp.exp(lookup['common_log_scale'])*(lookup['common_shape']+
            sp.Rational(1,4)*sum(lookup[f'{t}_{r}'] for t in ('geo','phys') for r in range(1,5))))
frozen=sp.sympify(locked['locked_model_formula'],locals=symbols,evaluate=False)
a,b=evaluate(factorised),evaluate(frozen)
assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
assert np.allclose(a,b,atol=1e-9,rtol=1e-9)
checks.append({'component':'complete_factorised_primary', 'max_absolute_transcription_difference':float(np.max(np.abs(a-b)))})
output=HERE/'Appendix_Frozen_Model_Expressions.tex'
tex.append('\n'+r'\endgroup'+'\n')
output.write_text(''.join(tex))
(HERE/'frozen_formula_components.json').write_text(json.dumps({x['label']:x['raw'] for x in components},indent=2))
(HERE/'formula_transcription_checks.json').write_text(json.dumps({
    'purpose':'Transcription check only; no training, model selection or final-test reevaluation.',
    'sample':'4096 deterministic rows from stored rotation-1 training discovery features',
    'checks':checks,
    'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
},indent=2))
with ZipFile('/Users/novwin/Downloads/FE results preparation.docx') as z:
    (HERE/'fe_model_geometry_and_input_fields.png').write_bytes(z.read('word/media/image6.png'))
print(json.dumps({'appendix':str(output),'components':len(components),'checks':checks},indent=2))
