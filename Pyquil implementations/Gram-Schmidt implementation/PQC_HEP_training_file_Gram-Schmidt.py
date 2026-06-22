"""
PQC_HEP_training_file_Gram-Schmidt.py

Author: Adriano Pinto Claro da Fonseca
Email: apcf@topfonseca.com
Institutional email: adriano.fonseca@tecnico.ulisboa.pt
Affiliation: Student at Instituto Superior Técnico, Universidade de Lisboa

Circuit templates and numbering follow Figure 2 of Sim et al. (2019) — see references in README.md.

Usage:
    python PQC_HEP_training_file_Gram-Schmidt.py
    python PQC_HEP_training_file_Gram-Schmidt.py --circuit-number 5 --batch-size 50

    All arguments are optional; defaults come from USER SETTINGS below.

Arguments (all optional):
    --circuits-file  : path to the circuits library Python file
    --circuit-number : integer 1-19, circuit label from the source paper
    --n-layers       : integer >= 1, number of repeating layer applications
    --n-iter         : integer >= 1, number of full passes over all events
    --batch-size     : events per mini-batch (must evenly divide total events used)
    --data-folder    : folder containing signal and background CSV files
    --sig            : signal filename (relative to --data-folder) or full path
    --bkg            : background filename (relative to --data-folder) or full path
    --max-events     : maximum total events to use

    Data file format (CSV, one event per line):
        col 0-11 : 12 floating-point feature values
        col 12   : event weight (float)
        col 13   : process name (string), ignored

Amplitude embedding:
    The normalised 16-dimensional amplitude vector becomes the first column of a
    16x16 unitary U. The remaining 15 columns are filled by iterative Gram-Schmidt
    orthogonalisation using random complex Gaussian vectors. Only U[:,0] is ever
    reached by the circuit (which starts from |0000>), so the random filler
    columns do not affect any measurement outcome.

Training pipeline (mini-batch mode):
    Embeddings are pre-computed once per event. Events are divided into batches
    of BATCH_SIZE. One complete sweep through all batches is one N_iter pass.

    Per batch:
      1. Run all BATCH_SIZE embedded circuits at current theta -> P(|1>) per event.
      2. Compute the batch loss (weighted binary cross-entropy, sum is over batch events only).
      3. Compute the batch gradient via the parameter shift rule.
      4. Take one Adam step (sklearn AdamOptimizer) -> update theta.

    Adam optimizer: sklearn AdamOptimizer, standard constant-beta EMA formulation.

Gradient assumption:
    The parameter shift rule gives the mathematically exact gradient ONLY for specific
    gate structures. The code classifies each theta parameter automatically before
    training and applies the correct rule per parameter:
      two_term  : plain RX(θ), RY(θ), RZ(θ) — Pauli generator, eigenvalues {+1,-1}.
                  Exact gradient: [f(θ+π/2) − f(θ−π/2)] / 2.
      four_term : CONTROLLED RX(θ)/RY(θ)/RZ(θ) — generator eigenvalues {-1,0,+1}.
                  Exact gradient: d1·[f(θ+π/2)−f(θ−π/2)] − d2·[f(θ+π)−f(θ−π)]
                  where d1=1/2, d2=(√2−1)/4.
    If a parameter feeds a gate with no known rule, a ValueError is raised before
    training begins — there is no silent failure.

Outputs (written to a timestamped folder):
    results.json   : full training record, loadable with json.load()
    classifier.py  : standalone classifier with trained parameters baked in

    Folder name encodes: circuit number, layers, N_iter, batch size, and timestamp.
    Example: circuit1_layers1_1000iter_batch100_2026-06-19_21-33-58/
"""

import sys
import time
import argparse
import os
import json
import importlib.util
import datetime
import textwrap
import numpy as np
from sklearn.metrics import log_loss
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer

from tqdm import tqdm
from pyquil import get_qc
from pyquil import Program
from pyquil.gates import RESET
from pyquil.quilbase import DefGate


# =============================================================================
# USER SETTINGS
# All adjustable parameters are collected here. Nothing else in this file
# should need to be changed between runs.
# =============================================================================

# Problem dimensions (tied to the data format and circuit library).
# Fixed by the physics setup; do not change without matching data and circuits.
N_QUBITS = 4
N_FEAT   = 12
N_DIM    = 2**N_QUBITS    # Hilbert-space dimension = 16

# Circuits and data.
# Selects the variational form circuit and specifies where to find the data files.
CIRCUITS_FILE   = "../circuits_library_PyQuil.py"  # path to the circuits library Python file
CIRCUIT_NUMBER  = 1                              # circuit template to use (integer 1-19)
N_LAYERS        = 1                              # number of variational block repetitions (>= 1)
DATA_FOLDER     = "../../Data"                      # folder containing the CSV files; None = current directory
SIG_FILE        = "train_sig.csv"               # signal CSV filename (relative to DATA_FOLDER)
BKG_FILE        = "train_bkg.csv"              # background CSV filename (relative to DATA_FOLDER)
MAX_EVENTS      = None                          # total events to use; None = use all events
BALANCE_CLASSES = False                          # if True, use equal S and B counts; requires MAX_EVENTS != None

# Event ordering.
# Controls the order and randomisation of training events.
# "random"           : signal and background events interleaved randomly (recommended)
# "signal_first"     : all signal events first, then all background
# "background_first" : all background events first, then all signal
EVENT_ORDER       = "random"
DATA_SHUFFLE_SEED = None     # seed for the initial event shuffle; None = random

# Between-pass reshuffling.
# If True, the event order is re-randomised at the start of each pass after pass 1.
# Requires EVENT_ORDER="random" (incompatible otherwise; caught at startup).
# Pass n uses seed (PASS_RESHUFFLE_SEED + n), giving reproducible but distinct
# shuffles each pass. If PASS_RESHUFFLE_SEED=None, each pass gets a fully
# random seed — runs will not be reproducible even with other seeds fixed.
RESHUFFLE_BETWEEN_PASSES = False
PASS_RESHUFFLE_SEED      = None   # base seed for between-pass reshuffles; None = random

# Training hyperparameters.
# Controls the training loop structure (epochs, batch size, shots) and the Adam
# optimiser step (learning rate and moment decay coefficients).
N_ITER     = 10        # number of full passes over all training events (epochs)

# Batch size.
# BATCH_SIZE_IS_FRACTION controls how BATCH_SIZE is interpreted:
#   False : BATCH_SIZE is the exact number of events per mini-batch (integer >= 1).
#           BATCH_SIZE = 1 means one event per gradient update (very noisy).
#   True  : BATCH_SIZE is a fraction of the total events used, in (0, 1].
#           BATCH_SIZE = 1.0 means the entire dataset in one batch (full batch GD).
#           The resolved event count must evenly divide the total number of events.
BATCH_SIZE_IS_FRACTION = True    # False = exact count, True = fraction of total events
BATCH_SIZE             = 0.1   # exact events per batch (int >= 1) or fraction (float in (0, 1])
Nm         = 1000  # shots per circuit evaluation
# Adam optimizer — sklearn AdamOptimizer, standard constant-beta EMA formulation.
# Uses constant beta_1/beta_2 as EMA decay coefficients.
ALPHA   = 5e-3    # learning rate
BETA1   = 0.9     # first-moment EMA decay coefficient (standard default)
BETA2   = 0.999   # second-moment EMA decay coefficient (standard default)
EPSILON = 1e-8    # numerical stability constant
THETA_INIT_SEED = None   # seed for the initial theta values; None = random

# Evaluation checkpoints.
# Specifies when to pause training and record the loss (during and after).
# Both lists accept: integers in [1, N_ITER], "start", "end".
# "start" = before any training. "end" = same as N_ITER.
#
# GLOBAL_EVAL_CHECKPOINTS : loss over ALL events (one forward pass per event).
# BATCH_EVAL_CHECKPOINTS  : pre-step loss from the LAST BATCH of that N_iter pass
#                           (no extra circuit runs; reuses the training forward pass).
GLOBAL_EVAL_CHECKPOINTS = ["start", "end"]
BATCH_EVAL_CHECKPOINTS  = ["start", "end"]

# Automatic step fill for evaluation checkpoints.
# When set to an integer >= 1, fills checkpoints from min to max of the integer
# set in steps. If only one integer exists (e.g. just "end"), starts from
# auto_step up to that endpoint. Example: ["start","end"] with N_ITER=50 and
# GLOBAL_EVAL_STEP=10 → auto-adds {10,20,30,40} → final set {10,20,30,40,50}.
GLOBAL_EVAL_STEP = None   # integer >= 1, or None to disable
BATCH_EVAL_STEP  = None   # integer >= 1, or None to disable

# QVM and Quilc server ports, and quantum computer selection.
# Port numbers for the running Docker containers (must match the docker run -p arguments).
QVM_PORT          = 5001        # host port mapped to the QVM container's internal port 5000
QUILC_PORT        = 5555        # host port mapped to the Quilc container's internal port 5555
# Name of the quantum computer to connect to via PyQuil.
#   "4q-qvm"     : local QVM simulator (default; requires Docker containers above).
#   "<QPU_NAME>" : a real Rigetti QPU (e.g., "Ankaa-2"); requires a Rigetti QCS account.
# Note: this implementation uses a custom 16x16 unitary gate (Gram-Schmidt) that cannot
# be directly executed on real hardware — real QPUs require native-gate decomposition first.
QUANTUM_COMPUTER  = "4q-qvm"
# Start containers:
#   docker run --rm -p <QVM_PORT>:5000   rigetti/qvm   -S
#   docker run --rm -p <QUILC_PORT>:5555 rigetti/quilc -R

# Timing.
SHOW_ELAPSED_TIME = True

# Output label.
# Optional text string prepended to the output folder name to help identify runs.
# None  : folder name is auto-generated as:
#         circuit{N}_layers{L}_{iter}iter_batch{batch}_{timestamp}
# string: folder name becomes {RUN_TAG}_circuit{N}_layers{L}_{iter}iter_batch{batch}_{timestamp}
RUN_TAG = None

# =============================================================================
# END OF USER SETTINGS
# =============================================================================
# SHOW_ELAPSED_TIME is validated here immediately so the timer can be started
# safely before the main validation block runs.
if not isinstance(SHOW_ELAPSED_TIME, bool):
    print("Error: SHOW_ELAPSED_TIME must be True or False.")
    sys.exit(1)
if SHOW_ELAPSED_TIME:
    _t_start = time.time()


# =============================================================================
# INPUT HANDLING
# =============================================================================
# Parses command-line arguments, validates all USER SETTINGS, resolves data
# file paths, loads the circuits library, and prepares evaluation checkpoints.


parser = argparse.ArgumentParser(
    description="PQC mini-batch training pipeline for HEP binary classification "
                "(Gram-Schmidt). All arguments optional; defaults from USER SETTINGS."
)
parser.add_argument("--circuits-file",  type=str, default=None, dest="circuits_file")
parser.add_argument("--circuit-number", type=int, default=None, dest="circuit_number")
parser.add_argument("--n-layers",       type=int, default=None, dest="n_layers")
parser.add_argument("--n-iter",         type=int, default=None, dest="n_iter")
parser.add_argument("--batch-size",     type=int, default=None, dest="batch_size")
parser.add_argument("--data-folder",    type=str, default=None, dest="data_folder")
parser.add_argument("--sig",            type=str, default=None, dest="sig_input")
parser.add_argument("--bkg",            type=str, default=None, dest="bkg_input")
parser.add_argument("--max-events",     type=int, default=None, dest="max_events")
args = parser.parse_args()

circuits_file  = args.circuits_file  if args.circuits_file  is not None else CIRCUITS_FILE
circuit_number = args.circuit_number if args.circuit_number is not None else CIRCUIT_NUMBER
n_layers       = args.n_layers       if args.n_layers       is not None else N_LAYERS
n_iter         = args.n_iter         if args.n_iter         is not None else N_ITER
_batch_from_cli = args.batch_size is not None
batch_size      = args.batch_size if _batch_from_cli else BATCH_SIZE
# When --batch-size is given on the command line, it is always an exact count;
# BATCH_SIZE_IS_FRACTION is ignored (CLI only accepts integers).
batch_size_is_fraction = (not _batch_from_cli) and BATCH_SIZE_IS_FRACTION
if _batch_from_cli and BATCH_SIZE_IS_FRACTION:
    print("Warning: --batch-size overrides BATCH_SIZE_IS_FRACTION=True. "
          "The CLI batch size is treated as an exact integer count.")
max_events     = args.max_events     if args.max_events     is not None else MAX_EVENTS

effective_data_folder = args.data_folder if args.data_folder is not None else DATA_FOLDER
effective_sig         = args.sig_input   if args.sig_input   is not None else SIG_FILE
effective_bkg         = args.bkg_input   if args.bkg_input   is not None else BKG_FILE

if effective_data_folder is not None:
    signal_file      = os.path.join(effective_data_folder, effective_sig)
    background_file  = os.path.join(effective_data_folder, effective_bkg)
    data_folder_name = os.path.basename(effective_data_folder.rstrip("/\\")) or effective_data_folder
else:
    signal_file      = effective_sig
    background_file  = effective_bkg
    data_folder_name = None

sig_name_only      = os.path.basename(signal_file)
bkg_name_only      = os.path.basename(background_file)
circuits_name_only = os.path.basename(circuits_file)

# Validate scalar settings.
if DATA_SHUFFLE_SEED is not None and not isinstance(DATA_SHUFFLE_SEED, int):
    print("Error: DATA_SHUFFLE_SEED must be an integer or None.")
    sys.exit(1)
if THETA_INIT_SEED is not None and not isinstance(THETA_INIT_SEED, int):
    print("Error: THETA_INIT_SEED must be an integer or None.")
    sys.exit(1)
if not isinstance(RESHUFFLE_BETWEEN_PASSES, bool):
    print("Error: RESHUFFLE_BETWEEN_PASSES must be True or False.")
    sys.exit(1)
if PASS_RESHUFFLE_SEED is not None and not isinstance(PASS_RESHUFFLE_SEED, int):
    print("Error: PASS_RESHUFFLE_SEED must be an integer or None.")
    sys.exit(1)
if EVENT_ORDER not in ("random", "signal_first", "background_first"):
    print("Error: EVENT_ORDER must be 'random', 'signal_first', or 'background_first'.")
    sys.exit(1)
if RESHUFFLE_BETWEEN_PASSES and EVENT_ORDER != "random":
    print(f"Error: RESHUFFLE_BETWEEN_PASSES=True conflicts with EVENT_ORDER='{EVENT_ORDER}'. "
          f"Reshuffling applies a fully random permutation that overrides EVENT_ORDER "
          f"from pass 2 onwards. Set EVENT_ORDER='random', or set RESHUFFLE_BETWEEN_PASSES=False.")
    sys.exit(1)
if not isinstance(BALANCE_CLASSES, bool):
    print("Error: BALANCE_CLASSES must be True or False.")
    sys.exit(1)
if BALANCE_CLASSES and max_events is None:
    print("Warning: BALANCE_CLASSES=True has no effect when MAX_EVENTS=None "
          "(all events are used regardless). Set MAX_EVENTS to the desired total "
          "if you want equal signal and background counts.")
if GLOBAL_EVAL_STEP is not None and (not isinstance(GLOBAL_EVAL_STEP, int) or GLOBAL_EVAL_STEP < 1):
    print("Error: GLOBAL_EVAL_STEP must be an integer >= 1 or None.")
    sys.exit(1)
if BATCH_EVAL_STEP is not None and (not isinstance(BATCH_EVAL_STEP, int) or BATCH_EVAL_STEP < 1):
    print("Error: BATCH_EVAL_STEP must be an integer >= 1 or None.")
    sys.exit(1)
if not isinstance(QUANTUM_COMPUTER, str):
    print("Error: QUANTUM_COMPUTER must be a string (e.g., '4q-qvm').")
    sys.exit(1)
if RUN_TAG is not None and not isinstance(RUN_TAG, str):
    print("Error: RUN_TAG must be a string or None.")
    sys.exit(1)
if n_layers   < 1:
    print("Error: n_layers must be >= 1.")
    sys.exit(1)
if n_iter     < 1:
    print("Error: n_iter must be >= 1.")
    sys.exit(1)
if not isinstance(BATCH_SIZE_IS_FRACTION, bool):
    print("Error: BATCH_SIZE_IS_FRACTION must be True or False.")
    sys.exit(1)
if batch_size_is_fraction:
    if not isinstance(batch_size, (int, float)) or not (0.0 < float(batch_size) <= 1.0):
        print(f"Error: BATCH_SIZE={batch_size} with BATCH_SIZE_IS_FRACTION=True must be in (0, 1].")
        sys.exit(1)
else:
    if not isinstance(batch_size, int) or batch_size < 1:
        print("Error: BATCH_SIZE must be an integer >= 1 when BATCH_SIZE_IS_FRACTION=False.")
        sys.exit(1)
if not isinstance(Nm, int) or Nm < 1:
    print("Error: Nm must be an integer >= 1.")
    sys.exit(1)
if not isinstance(ALPHA,   (int, float)) or ALPHA   <= 0:
    print("Error: ALPHA must be > 0.")
    sys.exit(1)
if not isinstance(BETA1,   (int, float)) or not (0 <= BETA1 < 1):
    print("Error: BETA1 must be in [0, 1).")
    sys.exit(1)
if not isinstance(BETA2,   (int, float)) or not (0 <= BETA2 < 1):
    print("Error: BETA2 must be in [0, 1).")
    sys.exit(1)
if not isinstance(EPSILON, (int, float)) or EPSILON <= 0:
    print("Error: EPSILON must be > 0.")
    sys.exit(1)
if not isinstance(QVM_PORT,   int) or not (1 <= QVM_PORT   <= 65535):
    print("Error: QVM_PORT must be in 1-65535.")
    sys.exit(1)
if not isinstance(QUILC_PORT, int) or not (1 <= QUILC_PORT <= 65535):
    print("Error: QUILC_PORT must be in 1-65535.")
    sys.exit(1)

# Load circuits library.
if not os.path.isfile(circuits_file):
    print(f"Error: circuits file '{circuits_file}' not found.")
    sys.exit(1)
spec = importlib.util.spec_from_file_location("circuits_library", circuits_file)
clib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clib)
if circuit_number < 1 or circuit_number >= len(clib.CIRCUITS) or clib.CIRCUITS[circuit_number] is None:
    print(f"Error: circuit_number {circuit_number} invalid. Available: 1-{len(clib.CIRCUITS)-1}.")
    sys.exit(1)
variational_form_fn, fixed_params, layer_params, circuit_description = clib.CIRCUITS[circuit_number]
N_PARAMS = fixed_params + n_layers * layer_params


def parse_checkpoints(raw, n_iter_total, setting_name, auto_step=None):
    """
    Convert a raw checkpoint list into a set of integer N_iter values and a
    flag indicating whether "start" was requested.
    Accepted values: integers in [1, n_iter_total], "start", "end".
    If auto_step is set, fills checkpoints from min to max of the integer set
    at that step interval. When only one integer is in the set (e.g. just "end"),
    filling starts from auto_step up to that single value.
    Returns (checkpoint_int_set, include_start).
    """
    if not raw:
        return set(), False
    ckpt_set      = set()
    include_start = False
    for val in raw:
        if val == "start":
            include_start = True
        elif val == "end":
            ckpt_set.add(n_iter_total)
        elif isinstance(val, int):
            if 1 <= val <= n_iter_total:
                ckpt_set.add(val)
            else:
                print(f"Warning: {setting_name} value {val} outside [1,{n_iter_total}], skipped.")
        else:
            print(f"Warning: {setting_name} value {repr(val)} not valid, skipped.")
    if auto_step is not None:
        if len(ckpt_set) == 0:
            print(f"Warning: {setting_name}_STEP={auto_step} is set but "
                  f"{setting_name} has no integer checkpoints to fill between; "
                  f"auto-fill has no effect.")
        else:
            before = len(ckpt_set)
            max_val = max(ckpt_set)
            min_val = min(ckpt_set) if len(ckpt_set) >= 2 else auto_step
            for v in range(min_val, max_val + 1, auto_step):
                if 1 <= v <= n_iter_total:
                    ckpt_set.add(v)
            if len(ckpt_set) == before:
                print(f"Warning: {setting_name}_STEP={auto_step} produced no "
                      f"additional checkpoints (step may be larger than the "
                      f"interval between existing checkpoints or N_ITER={n_iter_total}).")
    return ckpt_set, include_start

global_ckpt_set, global_do_start = parse_checkpoints(
    GLOBAL_EVAL_CHECKPOINTS, n_iter, "GLOBAL_EVAL_CHECKPOINTS", auto_step=GLOBAL_EVAL_STEP)
batch_ckpt_set,  batch_do_start  = parse_checkpoints(
    BATCH_EVAL_CHECKPOINTS,  n_iter, "BATCH_EVAL_CHECKPOINTS",  auto_step=BATCH_EVAL_STEP)

print(f"\n{'='*62}")
print(f"  EFFECTIVE CONFIGURATION")
print(f"{'='*62}")
print(f"  Circuits library  : {circuits_file}")
print(f"  Circuit           : [{circuit_number}] {circuit_description}")
print(f"  Layers / params   : {n_layers} layers, {N_PARAMS} parameters "
      f"({fixed_params} fixed + {n_layers}x{layer_params}/layer)")
print(f"  Signal file       : {signal_file}")
print(f"  Background file   : {background_file}")
print(f"  Max events        : {max_events if max_events is not None else 'all'}")
print(f"  Balance classes   : {BALANCE_CLASSES}")
print(f"  Event order       : {EVENT_ORDER}  (shuffle seed={DATA_SHUFFLE_SEED})")
print(f"  Reshuffle passes  : {RESHUFFLE_BETWEEN_PASSES}  (seed={PASS_RESHUFFLE_SEED})")
print(f"  N_iter            : {n_iter}")
_bs_display = (f"{batch_size} (= {float(batch_size)*100:.1f}% of total events; resolved after data loading)"
               if batch_size_is_fraction else str(batch_size))
print(f"  Batch size        : {_bs_display}")
print(f"  Nm (shots)        : {Nm}")
print(f"  Adam (sklearn)    : alpha={ALPHA}  beta1={BETA1}  beta2={BETA2}  epsilon={EPSILON}")
print(f"  Adam formulation  : sklearn AdamOptimizer, Kingma & Ba 2014 (constant-beta EMA)")
print(f"  Theta init seed   : {THETA_INIT_SEED}")
_gstep_str = f"  (auto-step every {GLOBAL_EVAL_STEP})" if GLOBAL_EVAL_STEP else ""
_bstep_str = f"  (auto-step every {BATCH_EVAL_STEP})"  if BATCH_EVAL_STEP  else ""
print(f"  Global eval ckpts : start={global_do_start}, at_n_iter={sorted(global_ckpt_set)}{_gstep_str}")
print(f"  Batch  eval ckpts : start={batch_do_start},  at_n_iter={sorted(batch_ckpt_set)}{_bstep_str}")
print(f"  Quantum computer  : {QUANTUM_COMPUTER}")
print(f"  QVM port          : {QVM_PORT}   Quilc port: {QUILC_PORT}")
if RUN_TAG is not None:
    print(f"  Run tag           : {RUN_TAG}")
print(f"{'='*62}\n")


# =============================================================================
# DATA LOADING
# =============================================================================
# Reads signal and background CSV files, normalises physics weights, selects
# and orders events, and validates that batch_size evenly divides the count.


def load_events(filepath, label):
    """
    Read one CSV file and return (features, weights, labels).
    Cols 0-11: float features. Col 12: float weight. Col 13: ignored.
    Malformed lines are skipped silently.
    """
    if not os.path.isfile(filepath):
        print(f"Error: file '{filepath}' not found.")
        sys.exit(1)
    feats   = []
    wts     = []
    skipped = 0
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            cols = line.split(',')
            try:
                feats.append([float(x) for x in cols[:12]])
                wts.append(float(cols[12]))
            except (IndexError, ValueError):
                skipped += 1
    if skipped:
        print(f"  Skipped {skipped} non-event lines in '{filepath}'")
    return feats, wts, [label] * len(feats)

print("Loading data...")
s_feats, s_weights, s_labels = load_events(signal_file,     1)
b_feats, b_weights, b_labels = load_events(background_file, 0)
s_weights = np.array(s_weights, dtype=float)
b_weights = np.array(b_weights, dtype=float)
if s_weights.sum() == 0:
    print("Error: signal weight sum is zero.")
    sys.exit(1)
if b_weights.sum() == 0:
    print("Error: background weight sum is zero.")
    sys.exit(1)
print(f"  Loaded : {len(s_feats) + len(b_feats)} events ({len(s_feats)} signal, {len(b_feats)} background)")

all_feats   = np.array(s_feats + b_feats)
all_weights = np.concatenate([s_weights, b_weights])
all_labels  = np.array(s_labels + b_labels, dtype=float)
rng         = np.random.default_rng(seed=DATA_SHUFFLE_SEED)

if BALANCE_CLASSES and max_events is not None:
    n_each    = max_events // 2
    available = min(len(s_feats), len(b_feats))
    if n_each > available:
        smaller = 'signal' if len(s_feats) <= len(b_feats) else 'background'
        print(f"Error: BALANCE_CLASSES needs {n_each}/class but only "
              f"{available} {smaller} available.")
        sys.exit(1)
    s_idx  = rng.permutation(len(s_feats))[:n_each]
    b_idx  = rng.permutation(len(b_feats))[:n_each]
    sel_f  = np.concatenate([np.array(s_feats)[s_idx], np.array(b_feats)[b_idx]])
    sel_w  = np.concatenate([s_weights[s_idx], b_weights[b_idx]])
    sel_l  = np.concatenate([np.array(s_labels, dtype=float)[s_idx],
                              np.array(b_labels, dtype=float)[b_idx]])
    order   = rng.permutation(len(sel_f))
    events  = sel_f[order]
    weights = sel_w[order]
    labels  = sel_l[order]
else:
    if max_events is not None and max_events > len(all_feats):
        print(f"Error: MAX_EVENTS={max_events} exceeds {len(all_feats)} available.")
        sys.exit(1)
    order   = rng.permutation(len(all_feats))
    events  = all_feats[order]
    weights = all_weights[order]
    labels  = all_labels[order]
    if max_events is not None:
        events  = events[:max_events]
        weights = weights[:max_events]
        labels  = labels[:max_events]

if EVENT_ORDER == "signal_first":
    sig_m = labels == 1
    bkg_m = labels == 0
    events  = np.concatenate([events[sig_m],  events[bkg_m]])
    weights = np.concatenate([weights[sig_m], weights[bkg_m]])
    labels  = np.concatenate([labels[sig_m],  labels[bkg_m]])
elif EVENT_ORDER == "background_first":
    sig_m = labels == 1
    bkg_m = labels == 0
    events  = np.concatenate([events[bkg_m],  events[sig_m]])
    weights = np.concatenate([weights[bkg_m], weights[sig_m]])
    labels  = np.concatenate([labels[bkg_m],  labels[sig_m]])

N_EVENTS = len(events)

# Normalise weights: sum(w_S) = sum(w_B) = N(background events in training set).
# Done after selection so the balance is correct for the events actually trained on.
_sig_mask    = labels == 1
_bkg_mask    = labels == 0
_n_bkg_train = int(_bkg_mask.sum())
if _n_bkg_train > 0 and _sig_mask.sum() > 0:
    weights[_sig_mask] *= _n_bkg_train / np.sum(weights[_sig_mask])
    weights[_bkg_mask] *= _n_bkg_train / np.sum(weights[_bkg_mask])

print(f"  Using  : {N_EVENTS} events ({int(_sig_mask.sum())} signal, {_n_bkg_train} background)")
print(f"  Weight normalisation: sum(w_S) = sum(w_B) = {_n_bkg_train}")

# Resolve BATCH_SIZE fraction to an integer now that N_EVENTS is known.
if batch_size_is_fraction:
    _resolved = int(round(float(batch_size) * N_EVENTS))
    if _resolved < 1:
        print(f"Error: BATCH_SIZE={batch_size} resolves to {_resolved} events for {N_EVENTS} total, "
              f"which is < 1. Use a larger fraction.")
        sys.exit(1)
    batch_size = _resolved
    print(f"  Batch size (resolved) : {batch_size}  (= {batch_size/N_EVENTS*100:.1f}% of {N_EVENTS} events)")

# Validate that batch_size evenly divides the event count.
if N_EVENTS % batch_size != 0:
    divisors = [i for i in range(1, N_EVENTS + 1) if N_EVENTS % i == 0]
    print(f"\nError: batch_size={batch_size} does not evenly divide {N_EVENTS} events.")
    if batch_size_is_fraction:
        valid_fracs = [f"{d}/{N_EVENTS} = {d/N_EVENTS:.4f}" for d in divisors if 0 < d < N_EVENTS]
        print(f"  Valid batch sizes: {divisors}")
        print(f"  Valid fractions: {valid_fracs[:12]}")
    else:
        print(f"  Valid batch sizes for {N_EVENTS} events: {divisors}")
    sys.exit(1)

N_BATCHES_PER_ITER = N_EVENTS // batch_size
print(f"  Batches per N_iter : {N_BATCHES_PER_ITER}  (batch_size={batch_size})\n")


# =============================================================================
# QUANTUM PIPELINE
# =============================================================================
# Defines all quantum operations: Gram-Schmidt amplitude embedding (16x16 unitary
# registered as a PyQuil custom gate), measurement, loss, and gradient via the
# parameter shift rule.


# --- QVM connection ----------------------------------------------------------

os.environ["QCS_SETTINGS_APPLICATIONS_QVM_URL"]   = f"http://127.0.0.1:{QVM_PORT}"
os.environ["QCS_SETTINGS_APPLICATIONS_QUILC_URL"] = f"tcp://127.0.0.1:{QUILC_PORT}"
qvm = get_qc(QUANTUM_COMPUTER)


# --- Preprocessing -----------------------------------------------------------

def preprocess_event(x_raw):
    """Pad to length 16 and normalise to unit length."""
    x_padded = np.concatenate([x_raw, np.zeros(N_DIM - N_FEAT)])
    norm = np.linalg.norm(x_padded)
    if norm < 1e-12:
        raise ValueError("Feature vector has zero norm.")
    return x_padded / norm


# --- Amplitude embedding via Gram-Schmidt ------------------------------------

def build_embedding_unitary(amplitudes):
    """
    Build a 16x16 unitary U whose first column equals amplitudes.
    The remaining 15 columns are orthonormalised random complex Gaussian vectors
    (Gram-Schmidt). Only U[:,0] matters physically because the circuit always
    starts from |0000> = e_0, so U|0000> = U[:,0] = amplitudes.
    """
    n     = len(amplitudes)
    basis = [amplitudes.astype(complex)]
    discards = 0
    while len(basis) < n:
        candidate = np.random.randn(n) + 1j * np.random.randn(n)
        for b in basis:
            candidate -= np.vdot(b, candidate) * b
        norm = np.linalg.norm(candidate)
        if norm > 1e-10:
            basis.append(candidate / norm)
            discards = 0
        else:
            discards += 1
            if discards >= 10_000:
                raise RuntimeError(
                    f"Gram-Schmidt failed: {discards} consecutive near-zero candidates "
                    f"while building column {len(basis)} of {n}."
                )
    return np.column_stack(basis)


def embed_state_in_pyquil(amplitudes):
    """
    Return (gate_def, embed_instr) for the Gram-Schmidt embedding of amplitudes.
    The gate is applied with reversed qubit order so qubit 0 is the LSB,
    matching the standard convention for the classification measurement.
    """
    unitary    = build_embedding_unitary(amplitudes)
    gate_def   = DefGate("AMPLITUDE_EMBED", unitary)
    EMBED_GATE = gate_def.get_constructor()
    embed_instr = EMBED_GATE(*range(N_QUBITS - 1, -1, -1))
    return gate_def, embed_instr


# --- Circuit measurement -----------------------------------------------------

def run_circuit_and_measure(gate_def, embed_instr, theta):
    """
    Build a fresh PyQuil Program from the pre-computed embedding, apply the
    variational form at the given theta, run Nm shots, and return P(qubit_0 = |1>).
    """
    p = Program()
    p += gate_def
    p += RESET()
    p += embed_instr
    p  = variational_form_fn(p, theta, n_layers)
    p.wrap_in_numshots_loop(Nm)
    result = qvm.run(p.measure_all()).get_register_map()["ro"]
    return float(np.array(result)[:, 0].mean())


# --- Loss function -----------------------------------------------------------

def compute_loss(p_values, event_labels, event_weights):
    """
    Weighted binary cross-entropy:
      L_K = (1/N) * sum_i { w_i * [-z_i*ln(y_i) - (1-z_i)*ln(1-y_i)] }
    """
    N      = len(p_values)
    p_clip = np.clip(p_values, 1e-7, 1 - 1e-7)
    raw    = log_loss(event_labels, p_clip, normalize=False, sample_weight=event_weights, labels=[0, 1])
    return float(raw / N)


# --- Shift rule classification ------------------------------------------------

def classify_shift_rules_pyquil(circuit_fn, n_params, n_layers):
    """
    Probe the variational form circuit with unique sentinel theta values and
    classify each parameter by which shift rule its gate requires.
    Called once before training. Returns a list of length n_params:
    'two_term' or 'four_term'.
    Raises ValueError for any unrecognised gate type — no silent failure.

    two_term  : plain RX/RY/RZ gates, Pauli generator, eigenvalues {+1,-1}.
    four_term : CONTROLLED RX/RY/RZ gates, generator eigenvalues {-1,0,+1}.
    """
    from pyquil import Program
    from pyquil.quilbase import Gate

    TWO_TERM_NAMES = {'RX', 'RY', 'RZ'}

    probe_theta     = np.arange(1.0, n_params + 1.0)
    probe_prog      = Program()
    circuit_fn(probe_prog, probe_theta, n_layers)
    sentinel_to_idx = {float(i + 1): i for i in range(n_params)}

    param_to_rule = {}
    for instr in probe_prog.instructions:
        if not isinstance(instr, Gate) or not instr.params:
            continue
        for param_val in instr.params:
            try:
                val = float(param_val)
            except TypeError:
                try:
                    val = float(param_val.real)
                except (AttributeError, TypeError, ValueError):
                    continue
            if val not in sentinel_to_idx:
                continue
            idx = sentinel_to_idx[val]
            if instr.name in TWO_TERM_NAMES and not instr.modifiers:
                param_to_rule[idx] = 'two_term'
            elif instr.name in TWO_TERM_NAMES and 'CONTROLLED' in instr.modifiers:
                param_to_rule[idx] = 'four_term'
            else:
                raise ValueError(
                    f"Parameter index {idx} feeds gate '{instr.name}' "
                    f"(modifiers={instr.modifiers}), which has no known "
                    f"parameter-shift rule. Add its rule to "
                    f"classify_shift_rules_pyquil before using it."
                )

    if len(param_to_rule) != n_params:
        missing = [i for i in range(n_params) if i not in param_to_rule]
        raise ValueError(
            f"Parameter indices {missing} were not matched to any gate. "
            f"Check that circuit_fn uses all parameters in theta[0..{n_params-1}]."
        )

    return [param_to_rule[i] for i in range(n_params)]


# --- Batch gradient via parameter shift rule ---------------------------------

def compute_batch_gradient_and_loss(batch_embeddings, batch_labels, batch_weights, theta,
                                    shift_rules):
    """
    Compute the gradient of the batch loss w.r.t. theta and the batch loss.

    Forward pass gives p_current for all batch events (also used for the loss).
    Then for each theta component k, the correct exact shift rule is applied:
      two_term  : grad[k] = (1/N) * dot(dL_dp, (p+ - p-) / 2)
                  p± evaluated at theta ± π/2.
      four_term : grad[k] = (1/N) * dot(dL_dp, d1*(fa+−fa−) − d2*(fb+−fb−))
                  a-shifts ±π/2, b-shifts ±π, d1=1/2, d2=(√2−1)/4.
    Total circuit evaluations: (1 + 2*n_two + 4*n_four) * batch_size.
    Returns (gradient_array, batch_loss_scalar) — loss is pre-step.
    """
    N  = len(batch_embeddings)
    D1 = 0.5
    D2 = (np.sqrt(2) - 1) / 4

    p_current = np.array([
        run_circuit_and_measure(gd, ei, theta)
        for gd, ei in batch_embeddings
    ])

    batch_loss = compute_loss(p_current, batch_labels, batch_weights)

    p_clip = np.clip(p_current, 1e-7, 1 - 1e-7)
    dL_dp  = batch_weights * (
        -batch_labels / p_clip + (1 - batch_labels) / (1 - p_clip)
    )

    gradients = np.zeros(len(theta))
    for k in range(len(theta)):
        if shift_rules[k] == 'two_term':
            tp    = theta.copy()
            tp[k] += np.pi / 2
            tm    = theta.copy()
            tm[k] -= np.pi / 2
            p_plus  = np.array([run_circuit_and_measure(gd, ei, tp) for gd, ei in batch_embeddings])
            p_minus = np.array([run_circuit_and_measure(gd, ei, tm) for gd, ei in batch_embeddings])
            dpdtheta = (p_plus - p_minus) / 2.0
        else:  # four_term
            ta_p    = theta.copy(); ta_p[k] += np.pi / 2
            ta_m    = theta.copy(); ta_m[k] -= np.pi / 2
            tb_p    = theta.copy(); tb_p[k] += np.pi
            tb_m    = theta.copy(); tb_m[k] -= np.pi
            f_ap = np.array([run_circuit_and_measure(gd, ei, ta_p) for gd, ei in batch_embeddings])
            f_am = np.array([run_circuit_and_measure(gd, ei, ta_m) for gd, ei in batch_embeddings])
            f_bp = np.array([run_circuit_and_measure(gd, ei, tb_p) for gd, ei in batch_embeddings])
            f_bm = np.array([run_circuit_and_measure(gd, ei, tb_m) for gd, ei in batch_embeddings])
            dpdtheta = D1 * (f_ap - f_am) - D2 * (f_bp - f_bm)

        gradients[k] = float(np.dot(dL_dp, dpdtheta) / N)

    return gradients, batch_loss


# --- Global evaluation -------------------------------------------------------

def evaluate_global_loss(all_embeddings, event_labels, event_weights, theta, desc="Eval"):
    """Run all events at given theta and return the global loss."""
    p_values = np.array([
        run_circuit_and_measure(gd, ei, theta)
        for gd, ei in tqdm(all_embeddings, desc=f"  {desc}", unit="event", leave=False)
    ])
    return compute_loss(p_values, event_labels, event_weights)


# --- Output utilities --------------------------------------------------------

class CircuitRecorder:
    """
    Wraps a PyQuil Program to intercept and record gate instructions added by
    the variational form function. Used after training to capture the gate sequence with
    trained parameter values for the classifier output file.
    """
    def __init__(self, base_program):
        self._program = base_program
        self.gates    = []

    def __iadd__(self, instruction):
        self._program += instruction
        if hasattr(instruction, 'name') and hasattr(instruction, 'qubits'):
            try:
                self.gates.append({
                    'name'     : instruction.name,
                    'modifiers': [str(m) for m in instruction.modifiers],
                    'qubits'   : [q.index for q in instruction.qubits],
                    'params'   : [float(p) for p in instruction.params],
                })
            except Exception:
                pass
        return self

    def measure_all(self):
        return self._program.measure_all()


def record_final_circuit(theta_final):
    """Replay the variational form with theta_final to record the trained gate sequence."""
    dummy_amp    = np.zeros(N_DIM, dtype=complex)
    dummy_amp[0] = 1.0
    gate_def, embed_instr = embed_state_in_pyquil(dummy_amp)
    base_prog  = Program()
    base_prog += gate_def
    base_prog += RESET()
    base_prog += embed_instr
    recorder   = CircuitRecorder(base_prog)
    variational_form_fn(recorder, np.array(theta_final), n_layers)
    return recorder.gates


def generate_theta_labels(gate_sequence):
    """Return a human-readable label for each parameterised gate in the variational form."""
    out = []
    for gate in gate_sequence:
        if gate['params']:
            name    = gate['name']
            is_ctrl = 'CONTROLLED' in gate['modifiers']
            target  = gate['qubits'][-1]
            ctrl    = gate['qubits'][0] if is_ctrl else None
            val     = gate['params'][0]
            if is_ctrl:
                out.append(f"CTRL-{name}(ctrl=q{ctrl}, tgt=q{target}) = {val:.6f} rad")
            else:
                out.append(f"{name}(q{target}) = {val:.6f} rad")
    return out


def gate_to_pyquil_line(gate):
    """Convert a recorded gate dict to one line of PyQuil Python code."""
    name      = gate['name']
    modifiers = gate['modifiers']
    qubits    = gate['qubits']
    params    = gate['params']
    is_ctrl   = 'CONTROLLED' in modifiers
    if name == 'CNOT' and not is_ctrl:   return f"    p += CNOT({qubits[0]}, {qubits[1]})"
    elif name == 'CZ' and not is_ctrl:   return f"    p += CZ({qubits[0]}, {qubits[1]})"
    elif name == 'H' and not is_ctrl:    return f"    p += H({qubits[0]})"
    elif is_ctrl and params:             return f"    p += {name}({params[0]:.10f}, {qubits[-1]}).controlled({qubits[0]})"
    elif params:                         return f"    p += {name}({params[0]:.10f}, {qubits[0]})"
    else:                                return f"    # Unrecognised gate {name} on {qubits}"


# =============================================================================
# TRAINING SETUP
# =============================================================================
# Pre-computes one Gram-Schmidt embedding per event (16x16 unitary), initialises
# theta, and creates the sklearn Adam optimiser.


# Pre-compute one Gram-Schmidt embedding per event (16x16 unitary + instruction).
# All N_iter passes reuse these cached embeddings, eliminating repeated construction.
print(f"Pre-computing Gram-Schmidt embeddings ({N_EVENTS} events)...")
all_embeddings = []
for x_raw in tqdm(events, desc="Embedding", unit="event", leave=True):
    amplitudes = preprocess_event(x_raw)
    all_embeddings.append(embed_state_in_pyquil(amplitudes))
print()

# Classify each theta parameter by the shift rule its gate requires.
shift_rules = classify_shift_rules_pyquil(variational_form_fn, N_PARAMS, n_layers)
_n_two  = shift_rules.count('two_term')
_n_four = shift_rules.count('four_term')
print(f"  Shift rule classification: {_n_two} two-term (RX/RY/RZ), "
      f"{_n_four} four-term (CONTROLLED RX/RY/RZ)\n")

# Initialise theta uniformly in [-pi, pi].
rng_theta     = np.random.default_rng(seed=THETA_INIT_SEED)
theta_current = rng_theta.uniform(-np.pi, np.pi, N_PARAMS)

# Initialise sklearn AdamOptimizer.
# params is a list containing theta_current; update_params() modifies it in place.
optimizer = AdamOptimizer(
    params             = [theta_current],
    learning_rate_init = ALPHA,
    beta_1             = BETA1,
    beta_2             = BETA2,
    epsilon            = EPSILON,
)

# Storage for checkpoint results.
global_eval_results = {}   # key: "start" or str(n_iter) -> loss value (float)
batch_eval_results  = {}   # key: "start" or str(n_iter) -> loss value (float)


n_width     = len(str(n_iter))
label_width = max(5, n_width)   # "start" is 5 chars; ensures [n=start] aligns with [n=  N]

print(f"Starting training: {n_iter} passes x {N_BATCHES_PER_ITER} batches x {batch_size} events/batch")
print(f"  Adam : sklearn AdamOptimizer, Kingma & Ba 2014  alpha={ALPHA}  beta1={BETA1}  beta2={BETA2}\n")


# =============================================================================
# PRE-TRAINING EVALUATIONS
# =============================================================================

if global_do_start or batch_do_start:
    _b_start = "--------"
    _g_start = "--------"
    if global_do_start:
        loss = evaluate_global_loss(all_embeddings, labels, weights, theta_current, desc="  Global [start]")
        global_eval_results["start"] = float(loss)
        _g_start = f"{loss:.6f}"
    if batch_do_start:
        p_start = np.array([
            run_circuit_and_measure(gd, ei, theta_current)
            for gd, ei in tqdm(all_embeddings[:batch_size],
                                desc="  Batch  [start]", unit="event", leave=False)
        ])
        loss_start = compute_loss(p_start, labels[:batch_size], weights[:batch_size])
        batch_eval_results["start"] = float(loss_start)
        _b_start = f"{loss_start:.6f}"
    print(f"  [n={'start':>{label_width}}]  batch = {_b_start}   GLOBAL = {_g_start}")
    print()


# =============================================================================
# TRAINING LOOP
# =============================================================================
# Runs N_iter passes over all events. Each pass sweeps through N_BATCHES_PER_ITER
# mini-batches, computing one batch gradient and taking one Adam step per batch.


last_batch_loss = None

pass_bar = tqdm(total=n_iter, desc="Training", unit="pass", position=0, leave=True)

for current_iter in range(1, n_iter + 1):

    if RESHUFFLE_BETWEEN_PASSES and current_iter > 1:
        if PASS_RESHUFFLE_SEED is not None:
            rng_reshuffle = np.random.default_rng(seed=PASS_RESHUFFLE_SEED + current_iter)
        else:
            rng_reshuffle = np.random.default_rng()
        perm           = rng_reshuffle.permutation(N_EVENTS)
        pass_embeddings = [all_embeddings[i] for i in perm]
        pass_labels     = labels[perm]
        pass_weights    = weights[perm]
    else:
        pass_embeddings = all_embeddings
        pass_labels     = labels
        pass_weights    = weights

    with tqdm(range(N_BATCHES_PER_ITER),
              desc=f"  Pass {current_iter:{n_width}d}/{n_iter}",
              unit="batch", leave=False, position=1) as batch_bar:
        for batch_idx in batch_bar:
            start = batch_idx * batch_size
            end   = start + batch_size

            b_embeddings = pass_embeddings[start:end]
            b_labels     = pass_labels[start:end]
            b_weights    = pass_weights[start:end]

            grad, current_batch_loss = compute_batch_gradient_and_loss(
                b_embeddings, b_labels, b_weights, theta_current, shift_rules
            )

            # Adam step.
            optimizer.update_params([theta_current], [grad])
            last_batch_loss = current_batch_loss

    pass_bar.update(1)

    if current_iter in batch_ckpt_set or current_iter in global_ckpt_set:
        _b_str = "--------"
        _g_str = "--------"
        if current_iter in batch_ckpt_set:
            batch_eval_results[str(current_iter)] = float(last_batch_loss)
            _b_str = f"{last_batch_loss:.6f}"
        if current_iter in global_ckpt_set:
            g_loss = evaluate_global_loss(
                all_embeddings, labels, weights, theta_current,
                desc=f"  Global [n={current_iter:{n_width}d}]"
            )
            global_eval_results[str(current_iter)] = float(g_loss)
            _g_str = f"{g_loss:.6f}"
        tqdm.write(
            f"  [n={current_iter:{label_width}d}]  batch = {_b_str}   GLOBAL = {_g_str}"
        )

pass_bar.close()
theta_final   = theta_current.tolist()
gate_sequence = record_final_circuit(theta_final)
theta_labels  = generate_theta_labels(gate_sequence)


# =============================================================================
# OUTPUT GENERATION
# =============================================================================
# Writes the results JSON and a standalone classifier script to a timestamped
# folder named after the circuit, layers, training settings, and run time.


timestamp  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_base_name = (f"circuit{circuit_number}_layers{n_layers}_"
              f"{n_iter}iter_batch{batch_size}_{timestamp}")
output_dir = f"{RUN_TAG}_{_base_name}" if RUN_TAG is not None else _base_name
os.makedirs(output_dir, exist_ok=True)
print(f"\nSaving outputs to: {output_dir}/")

json_path = os.path.join(output_dir, "results.json")
results = {
    "run_info": {
        "circuits_library"        : circuits_name_only,
        "signal_file"             : sig_name_only,
        "background_file"         : bkg_name_only,
        "data_folder"             : data_folder_name,
        "timestamp"               : timestamp,
        "n_signal"                : len(s_feats),
        "n_background"            : len(b_feats),
        "n_events_loaded"         : len(all_feats),
        "n_events_used"           : N_EVENTS,
        "max_events_limit"        : max_events,
        "BALANCE_CLASSES"         : BALANCE_CLASSES,
        "EVENT_ORDER"             : EVENT_ORDER,
        "DATA_SHUFFLE_SEED"       : DATA_SHUFFLE_SEED,
        "RESHUFFLE_BETWEEN_PASSES": RESHUFFLE_BETWEEN_PASSES,
        "PASS_RESHUFFLE_SEED"     : PASS_RESHUFFLE_SEED,
        "N_ITER"                  : n_iter,
        "BATCH_SIZE_IS_FRACTION"  : BATCH_SIZE_IS_FRACTION,
        "BATCH_SIZE"              : batch_size,
        "N_BATCHES_PER_ITER"      : N_BATCHES_PER_ITER,
        "Nm"                      : Nm,
        "ALPHA"                   : ALPHA,
        "BETA1"                   : BETA1,
        "BETA2"                   : BETA2,
        "EPSILON"                 : EPSILON,
        "adam_formulation"        : "sklearn_AdamOptimizer_Kingma_Ba_2014",
        "THETA_INIT_SEED"         : THETA_INIT_SEED,
        "GLOBAL_EVAL_CHECKPOINTS" : sorted(global_ckpt_set),
        "GLOBAL_EVAL_STEP"        : GLOBAL_EVAL_STEP,
        "BATCH_EVAL_CHECKPOINTS"  : sorted(batch_ckpt_set),
        "BATCH_EVAL_STEP"         : BATCH_EVAL_STEP,
        "QUANTUM_COMPUTER"        : QUANTUM_COMPUTER,
        "QVM_PORT"                : QVM_PORT,
        "QUILC_PORT"              : QUILC_PORT,
        "RUN_TAG"                 : RUN_TAG,
    },
    "circuit": {
        "circuit_number"     : circuit_number,
        "circuit_description": circuit_description,
        "n_layers"           : n_layers,
        "fixed_params"       : fixed_params,
        "layer_params"       : layer_params,
        "total_params"       : N_PARAMS,
    },
    "output": {
        "theta_final"            : theta_final,
        "theta_labels"           : theta_labels,
        "global_eval_checkpoints": global_eval_results,
        "batch_eval_checkpoints" : batch_eval_results,
    },
}
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  [1/2] Results JSON : {json_path}")

gate_lines     = "\n".join(gate_to_pyquil_line(g) for g in gate_sequence)
# Add one extra level of indentation so gate_lines land inside classify() after textwrap.dedent.
gate_lines_cls = textwrap.indent(gate_lines, "    ")
theta_literal  = repr(theta_final)
labels_literal = repr(theta_labels)

classifier_script = textwrap.dedent(f"""\
    # classifier.py  —  standalone classifier generated by PQC_HEP_training_file_Gram-Schmidt.py
    #
    # Training info (read-only, do not modify):
    #   Timestamp  : {timestamp}
    #   Circuit    : [{circuit_number}] {circuit_description}
    #   Layers     : {n_layers}  |  Parameters: {N_PARAMS}
    #   Events     : {N_EVENTS}  |  N_iter: {n_iter}  |  Batch size: {batch_size}
    #   Simulation : shot-based  Nm={Nm}
    #   Adam       : alpha={ALPHA}  beta1={BETA1}  beta2={BETA2}
    #                epsilon={EPSILON}  theta_init_seed={THETA_INIT_SEED}
    #   QVM target : {QUANTUM_COMPUTER}
    #
    # Note: the Gram-Schmidt embedding uses a custom 16x16 DefGate.
    #       This requires the PyQuil QVM — it cannot run on real Rigetti hardware.
    #       Start the containers before running:
    #         docker run --rm -p {QVM_PORT}:5000   rigetti/qvm   -S
    #         docker run --rm -p {QUILC_PORT}:5555 rigetti/quilc -R
    #
    # Usage:
    #   python classifier.py              — classify one event (set your_event_features below)
    #   python classifier.py events.csv  — classify all events in a CSV file
    #                                       (col 0-11: features; col 12+: ignored)
    #
    # Requirements: pyquil (QVM running), numpy

    import sys
    import csv
    import os
    import numpy as np
    from pyquil import get_qc, Program
    from pyquil.gates import RX, RY, RZ, H, CNOT, CZ, RESET
    from pyquil.quilbase import DefGate

    # ---- Inference settings baked in from training (do not modify) ---------------
    N_QUBITS   = {N_QUBITS}
    N_FEAT     = {N_FEAT}
    N_DIM      = {N_DIM}
    Nm         = {Nm}          # shots per circuit evaluation
    QVM_PORT   = {QVM_PORT}
    QUILC_PORT = {QUILC_PORT}

    # ---- Trained parameters (baked in as literals inside the circuit below) ------
    THETA_FINAL  = {theta_literal}
    THETA_LABELS = {labels_literal}

    def preprocess(x_raw):
        # Zero-pad 12 features to 16 dimensions and normalise to unit L2 norm.
        x_padded = np.concatenate([np.asarray(x_raw, dtype=float), np.zeros(N_DIM - N_FEAT)])
        norm = np.linalg.norm(x_padded)
        if norm < 1e-12:
            raise ValueError("Zero-norm feature vector — cannot embed.")
        return x_padded / norm

    def build_embedding_unitary(amplitudes):
        # Build a 16x16 unitary with amplitudes as first column via Gram-Schmidt orthogonalisation.
        # The remaining 15 columns are random orthonormal vectors (filler; they are never reached
        # by the circuit since the register starts in |0000>).
        n        = len(amplitudes)
        basis    = [amplitudes.astype(complex)]
        discards = 0
        while len(basis) < n:
            candidate = np.random.randn(n) + 1j * np.random.randn(n)
            for b in basis:
                candidate -= np.vdot(b, candidate) * b
            norm = np.linalg.norm(candidate)
            if norm > 1e-10:
                basis.append(candidate / norm)
                discards = 0
            else:
                discards += 1
                if discards >= 10_000:
                    raise RuntimeError("Gram-Schmidt orthogonalisation failed.")
        return np.column_stack(basis)

    def embed_state_in_pyquil(amplitudes):
        # Apply the 16x16 Gram-Schmidt unitary via a PyQuil DefGate, encoding the amplitude vector.
        unitary    = build_embedding_unitary(amplitudes)
        gate_def   = DefGate("AMPLITUDE_EMBED", unitary)
        EMBED_GATE = gate_def.get_constructor()
        p  = Program()
        p += gate_def
        p += RESET()
        p += EMBED_GATE(*range(N_QUBITS - 1, -1, -1))
        return p

    def classify(features, qvm):
        # Build the full PQC (embedding + variational form) and run Nm shots.
        # The trained rotation angles are baked into the gate lines below as literals.
        # Returns P(qubit 0 = |1>) estimated from measurement outcomes.
        amplitudes = preprocess(features)
        p          = embed_state_in_pyquil(amplitudes)
{gate_lines_cls}
        p.wrap_in_numshots_loop(Nm)
        result = qvm.run(p.measure_all()).get_register_map()["ro"]
        # Column 0 corresponds to qubit 0; average over Nm shots gives P(|1>).
        return float(np.array(result)[:, 0].mean())

    def print_result(y, index=None):
        # Print one result line: probability and SIGNAL/BACKGROUND label.
        prefix = f"Event {{index:6d}} : " if index is not None else ""
        label  = "SIGNAL" if y >= 0.5 else "BACKGROUND"
        print(f"{{prefix}}P(signal) = {{y:.4f}}   ->   {{label}}")

    # Connect to the QVM once and reuse for all events.
    os.environ["QCS_SETTINGS_APPLICATIONS_QVM_URL"]   = f"http://127.0.0.1:{{QVM_PORT}}"
    os.environ["QCS_SETTINGS_APPLICATIONS_QUILC_URL"] = f"tcp://127.0.0.1:{{QUILC_PORT}}"
    qvm = get_qc("{QUANTUM_COMPUTER}")

    if len(sys.argv) > 1:
        # Mode 2: read a CSV file and classify every event in it.
        # Expected CSV format: col 0-11 = 12 features; col 12+ ignored (same as training data).
        with open(sys.argv[1], newline='') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if not row or row[0].startswith('#'):
                    continue   # skip blank lines and comment lines
                print_result(classify([float(x) for x in row[:N_FEAT]], qvm), index=i)
    else:
        # Mode 1: classify a single event — replace your_event_features with real values.
        your_event_features = [0.0] * N_FEAT   # <-- replace with your 12 feature values
        print_result(classify(your_event_features, qvm))
""")

classifier_path = os.path.join(output_dir, "classifier.py")
with open(classifier_path, "w") as f:
    f.write(classifier_script)
print(f"  [2/2] Classifier   : {classifier_path}")

print(f"\nTraining complete.")
print(f"  Circuit      : [{circuit_number}] {circuit_description}")
print(f"  Layers       : {n_layers}  |  Parameters: {N_PARAMS}")
print(f"  Events       : {N_EVENTS} used of {len(all_feats)} loaded")
print(f"  N_iter       : {n_iter}  |  Batches/iter: {N_BATCHES_PER_ITER}  |  Batch size: {batch_size}")
print(f"  theta_final  : {np.round(theta_final, 4)}")
if global_eval_results:
    last_g = sorted((k for k in global_eval_results if k != "start"), key=int)
    if last_g: print(f"  Last global eval (n={last_g[-1]}): loss={global_eval_results[last_g[-1]]:.6f}")
if batch_eval_results:
    last_b = sorted((k for k in batch_eval_results if k != "start"), key=int)
    if last_b: print(f"  Last batch  eval (n={last_b[-1]}): loss={batch_eval_results[last_b[-1]]:.6f}")
print(f"  Output folder: {output_dir}/")

if SHOW_ELAPSED_TIME:
    _elapsed = time.time() - _t_start
    print(f"\n  Elapsed time : {_elapsed:.1f} s  ({_elapsed/60:.1f} min)")
    print(f"  Per event    : {_elapsed/N_EVENTS:.3f} s/event  (N_iter={n_iter}, Nm={Nm} shots, {N_EVENTS} events)")
