"""
PQC_HEP_training_file_Qiskit.py

Author: Adriano Pinto Claro da Fonseca
Email: apcf@topfonseca.com
Institutional email: adriano.fonseca@tecnico.ulisboa.pt
Affiliation: Student at Instituto Superior Técnico, Universidade de Lisboa

Circuit templates and numbering follow Figure 2 of Sim et al. (2019) — see references in README.md.

Usage:
    python PQC_HEP_training_file_Qiskit.py
    python PQC_HEP_training_file_Qiskit.py --circuit-number 5 --n-layers 2 --batch-size 50

    All arguments are optional; defaults come from USER SETTINGS below.
    Command-line arguments always override USER SETTINGS values.

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

Training pipeline (mini-batch mode):
    All circuits (embedding + variational form) are pre-built and transpiled once before
    training. Events are divided into batches of BATCH_SIZE. One complete sweep
    through all batches constitutes one N_iter pass (epoch).

    Per batch:
      1. Run all BATCH_SIZE circuits at the current theta -> P(|1>) per event.
      2. Compute the batch loss (weighted binary cross-entropy, sum is over batch events only).
      3. Compute the batch gradient via the parameter shift rule, accumulating
         contributions from all batch events.
      4. Take one Adam step (sklearn AdamOptimizer) -> update theta.

    After each complete N_iter pass:
      - BATCH_EVAL_CHECKPOINTS : store the pre-step loss of the last batch.
      - GLOBAL_EVAL_CHECKPOINTS: run ALL events with current theta, store loss.

    Adam optimizer: sklearn AdamOptimizer, standard constant-beta EMA formulation.

Gradient assumption:
    The parameter shift rule gives the mathematically exact gradient ONLY for specific
    gate structures. The code classifies each theta parameter automatically before
    training and applies the correct rule per parameter:
      two_term  : gates of the form exp(-iθ/2·P), P a Pauli, eigenvalues {+1,-1}.
                  Exact gradient: [f(θ+π/2) − f(θ−π/2)] / 2.
                  Satisfied by: Rx(θ), Ry(θ), Rz(θ).
      four_term : controlled-Pauli-rotation gates, generator eigenvalues {-1,0,+1}.
                  Exact gradient: d1·[f(θ+π/2)−f(θ−π/2)] − d2·[f(θ+π)−f(θ−π)]
                  where d1=1/2, d2=(√2−1)/4.
                  Satisfied by: CRx(θ), CRy(θ), CRz(θ).
    If a parameter feeds a gate with no known rule, a ValueError is raised before
    training begins — there is no silent failure.

Outputs (written to a timestamped folder):
    results.json   : full training record, loadable with json.load()
    classifier.py  : standalone classifier with trained parameters baked in

    Folder name encodes: circuit number, layers, N_iter, batch size, and timestamp.
    Example: circuit2_layers1_50iter_batch100_2026-06-19_21-33-58/
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
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
# Note: qiskit_ibm_runtime is NOT imported here unconditionally. If IBM_BACKEND
# is set in USER SETTINGS, the code will attempt to import QiskitRuntimeService
# from qiskit_ibm_runtime at runtime (inside the QUANTUM PIPELINE section).
# If qiskit_ibm_runtime is not installed and IBM_BACKEND is set, the code exits
# with a clear installation instruction. If IBM_BACKEND=None (default), the
# import is never attempted and qiskit_ibm_runtime does not need to be installed.


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
CIRCUITS_FILE   = "circuits_library_Qiskit.py"  # path to the circuits library Python file
CIRCUIT_NUMBER  = 1                              # circuit template to use (integer 1-19)
N_LAYERS        = 1                              # number of variational block repetitions (>= 1)
DATA_FOLDER     = "../Data"                      # folder containing the CSV files; None = current directory
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
# Controls the training loop structure (epochs, batch size) and the Adam
# optimiser step (learning rate and moment decay coefficients).
N_ITER = 10    # number of full passes over all training events (epochs)

# Batch size.
# BATCH_SIZE_IS_FRACTION controls how BATCH_SIZE is interpreted:
#   False : BATCH_SIZE is the exact number of events per mini-batch (integer >= 1).
#           BATCH_SIZE = 1 means one event per gradient update (very noisy).
#   True  : BATCH_SIZE is a fraction of the total events used, in (0, 1].
#           BATCH_SIZE = 1.0 means the entire dataset in one batch (full batch GD).
#           The resolved event count must evenly divide the total number of events.
BATCH_SIZE_IS_FRACTION = True   # False = exact count, True = fraction of total events
BATCH_SIZE             = 0.1   # exact events per batch (int >= 1) or fraction (float in (0, 1])

# Adam optimizer — sklearn AdamOptimizer, standard constant-beta EMA formulation.
# Uses constant beta_1/beta_2 as EMA decay coefficients.
ALPHA   = 5e-3    # learning rate
BETA1   = 0.9     # first-moment EMA decay coefficient (standard default)
BETA2   = 0.999   # second-moment EMA decay coefficient (standard default)
EPSILON = 1e-8    # numerical stability constant
THETA_INIT_SEED = None      # seed for the initial theta values; None = random

# Evaluation checkpoints.
# Specifies when to pause training and record the loss (during and after).
#
# Both lists accept any combination of:
#   integers in [1, N_ITER] : evaluate at the end of that N_iter pass
#   "start"                 : evaluate before any training (initial random theta)
#   "end"                   : equivalent to N_ITER (the last pass)
#
# Only N_iter-boundary values are valid for integers (no mid-pass snapshots).
# "start" is the only exception — it is not at an N_iter boundary.
#
# GLOBAL_EVAL_CHECKPOINTS
#   Computes the loss over ALL training events with the current theta.
#   Requires one forward-pass circuit run per event each time it triggers.
#   "start" uses the initial random theta over all events.
#
# BATCH_EVAL_CHECKPOINTS
#   Records the pre-step loss from the LAST BATCH of the specified N_iter pass.
#   No extra circuit runs needed — reuses the forward pass already done inside
#   the gradient calculation for that batch.
#   "start" records the first batch loss before any Adam step has been taken.
GLOBAL_EVAL_CHECKPOINTS = ["start", "end"]
BATCH_EVAL_CHECKPOINTS  = ["start", "end"]

# Automatic step fill for evaluation checkpoints.
# When set to an integer >= 1, automatically fills in checkpoints at regular
# intervals between the smallest and largest integer values already in the
# respective list. Example: list = [10, 50] with GLOBAL_EVAL_STEP = 10 →
# auto-adds {20, 30, 40}, giving final set {10, 20, 30, 40, 50}.
# The fill uses the step starting from the list's minimum upward; the first
# value that would exceed the list's maximum is not added.
# Set to None to disable automatic filling.
GLOBAL_EVAL_STEP = None   # integer >= 1, or None to disable
BATCH_EVAL_STEP  = None   # integer >= 1, or None to disable

# Simulation mode.
# Selects between exact statevector computation (no shot noise, deterministic)
# and shot-based measurement simulation (realistic for real quantum hardware).
# True  : exact P(|1>) from statevector — no shot noise, deterministic, fast.
# False : estimate P(|1>) from Nm measurement shots — realistic for real hardware.
USE_STATEVECTOR   = False
Nm                = 1000     # shots per circuit evaluation (ignored when USE_STATEVECTOR=True)
SIM_SEED          = None       # AerSimulator RNG seed (int >= 0) or None for a random seed
# Local simulation method (only used when IBM_BACKEND=None and USE_STATEVECTOR=False).
# "automatic"            : auto-select based on circuit (recommended; original behaviour).
# "statevector"          : exact internal state, then samples Nm shots (has shot noise).
#                          Different from USE_STATEVECTOR=True, which skips shot sampling.
# "matrix_product_state" : efficient for circuits with limited entanglement.
# "density_matrix"       : for open quantum systems with noise models.
SIMULATOR_METHOD  = "automatic"

# Real IBM Quantum hardware selection.
# Set IBM_BACKEND to an IBM QPU name to run on real hardware instead of local simulation.
#   None       : use the local AerSimulator (all settings above apply normally).
#   "ibm_..."  : real IBM QPU name, e.g. "ibm_brisbane", "ibm_kyoto" (requires account).
# Requires: pip install qiskit-ibm-runtime  and a valid IBM Quantum account.
# Set IBM_TOKEN to your API token, or None if credentials are saved in your environment.
# When IBM_BACKEND is set: USE_STATEVECTOR, SIMULATOR_METHOD, and SIM_SEED are ignored
# (real hardware always uses native shot-based execution).
IBM_BACKEND = None   # None = local AerSimulator; string = IBM QPU name for real hardware
IBM_TOKEN   = None   # IBM Quantum API token, or None if saved in the environment

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
    description="PQC mini-batch training pipeline for HEP binary classification (Qiskit). "
                "All arguments are optional; defaults come from USER SETTINGS."
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

# Command-line arguments override USER SETTINGS; None means "use USER SETTINGS".
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
if not isinstance(USE_STATEVECTOR, bool):
    print("Error: USE_STATEVECTOR must be True or False.")
    sys.exit(1)
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
if not isinstance(SIMULATOR_METHOD, str):
    print("Error: SIMULATOR_METHOD must be a string (e.g., 'automatic', 'statevector').")
    sys.exit(1)
if IBM_BACKEND is not None and not isinstance(IBM_BACKEND, str):
    print("Error: IBM_BACKEND must be a string (IBM QPU name) or None.")
    sys.exit(1)
if IBM_TOKEN is not None and not isinstance(IBM_TOKEN, str):
    print("Error: IBM_TOKEN must be a string or None.")
    sys.exit(1)
if IBM_BACKEND is not None and USE_STATEVECTOR:
    print("Error: IBM_BACKEND and USE_STATEVECTOR=True are incompatible. "
          "Real hardware always uses shot-based measurement.")
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
if max_events is not None and max_events < 1:
    print("Error: --max-events must be >= 1.")
    sys.exit(1)
if not USE_STATEVECTOR and (not isinstance(Nm, int) or Nm < 1):
    print("Error: Nm must be an integer >= 1 when USE_STATEVECTOR is False.")
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
if SIM_SEED is not None and (not isinstance(SIM_SEED, int) or SIM_SEED < 0):
    print("Error: SIM_SEED must be a non-negative integer or None.")
    sys.exit(1)

# Load circuits library.
if not os.path.isfile(circuits_file):
    print(f"Error: circuits file '{circuits_file}' not found.")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("circuits_library", circuits_file)
clib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clib)

if circuit_number < 1 or circuit_number >= len(clib.CIRCUITS) or clib.CIRCUITS[circuit_number] is None:
    print(f"Error: circuit_number {circuit_number} is invalid. "
          f"Available: 1 to {len(clib.CIRCUITS)-1}.")
    sys.exit(1)

variational_form_fn, fixed_params, layer_params, circuit_description = clib.CIRCUITS[circuit_number]
N_PARAMS = fixed_params + n_layers * layer_params


# Parse evaluation checkpoints.

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
                print(f"Warning: {setting_name} value {val} is outside "
                      f"[1, {n_iter_total}] and will be skipped.")
        else:
            print(f"Warning: {setting_name} value {repr(val)} is not valid and will be skipped.")

    # Auto-fill intermediate checkpoints at regular step intervals.
    # max is taken from the set; min is the set's minimum if 2+ values exist,
    # otherwise auto_step itself (handles the ["start","end"] case where only
    # one integer is present — fills from the first step up to that endpoint).
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
    GLOBAL_EVAL_CHECKPOINTS, n_iter, "GLOBAL_EVAL_CHECKPOINTS",
    auto_step=GLOBAL_EVAL_STEP)
batch_ckpt_set,  batch_do_start  = parse_checkpoints(
    BATCH_EVAL_CHECKPOINTS,  n_iter, "BATCH_EVAL_CHECKPOINTS",
    auto_step=BATCH_EVAL_STEP)


# Print effective configuration.

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
print(f"  Adam (sklearn)    : alpha={ALPHA}  beta1={BETA1}  beta2={BETA2}  epsilon={EPSILON}")
print(f"  Adam formulation  : sklearn AdamOptimizer, Kingma & Ba 2014 (constant-beta EMA)")
print(f"  Theta init seed   : {THETA_INIT_SEED}")
_gstep_str = f"  (auto-step every {GLOBAL_EVAL_STEP})" if GLOBAL_EVAL_STEP else ""
_bstep_str = f"  (auto-step every {BATCH_EVAL_STEP})"  if BATCH_EVAL_STEP  else ""
print(f"  Global eval ckpts : start={global_do_start}, at_n_iter={sorted(global_ckpt_set)}{_gstep_str}")
print(f"  Batch  eval ckpts : start={batch_do_start},  at_n_iter={sorted(batch_ckpt_set)}{_bstep_str}")
if USE_STATEVECTOR:
    _sim_desc = "Statevector (exact, no shot noise)"
else:
    _sim_desc = f"Shot-based  Nm={Nm}  method={SIMULATOR_METHOD}"
print(f"  Simulation mode   : {_sim_desc}  (sim seed={SIM_SEED if SIM_SEED is not None else 'random'})")
if IBM_BACKEND is not None:
    print(f"  IBM backend       : {IBM_BACKEND}  (real quantum hardware)")
if RUN_TAG is not None:
    print(f"  Run tag           : {RUN_TAG}")
print(f"{'='*62}\n")


# =============================================================================
# DATA LOADING
# =============================================================================
# Reads signal and background CSV files, selects and orders events, normalises
# weights relative to the selected training set, and validates batch_size.

def load_events(filepath, label):
    """
    Read one CSV file and return (features, weights, labels).
    Cols 0-11: float features. Col 12: float weight. Col 13: ignored.
    Malformed lines are skipped silently.
    """
    if not os.path.isfile(filepath):
        print(f"Error: file '{filepath}' not found.")
        sys.exit(1)
    feats, wts = [], []
    skipped = 0
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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

# Combine and shuffle all events.
all_feats   = np.array(s_feats + b_feats)
all_weights = np.concatenate([s_weights, b_weights])
all_labels  = np.array(s_labels + b_labels, dtype=float)
rng         = np.random.default_rng(seed=DATA_SHUFFLE_SEED)

if BALANCE_CLASSES and max_events is not None:
    n_each    = max_events // 2
    available = min(len(s_feats), len(b_feats))
    if n_each > available:
        smaller = 'signal' if len(s_feats) <= len(b_feats) else 'background'
        print(f"Error: BALANCE_CLASSES requested {n_each} per class but only "
              f"{available} {smaller} events available. "
              f"Set MAX_EVENTS <= {2*available} or BALANCE_CLASSES=False.")
        sys.exit(1)
    s_idx   = rng.permutation(len(s_feats))[:n_each]
    b_idx   = rng.permutation(len(b_feats))[:n_each]
    sel_f   = np.concatenate([np.array(s_feats)[s_idx],               np.array(b_feats)[b_idx]])
    sel_w   = np.concatenate([s_weights[s_idx],                       b_weights[b_idx]])
    sel_l   = np.concatenate([np.array(s_labels, dtype=float)[s_idx], np.array(b_labels, dtype=float)[b_idx]])
    order   = rng.permutation(len(sel_f))
    events  = sel_f[order]
    weights = sel_w[order]
    labels  = sel_l[order]
else:
    if max_events is not None and max_events > len(all_feats):
        print(f"Error: MAX_EVENTS={max_events} exceeds {len(all_feats)} available events.")
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
# Defines all quantum operations: amplitude embedding, transpiled circuit
# building, shot-based or statevector measurement, loss, and gradient via the
# parameter shift rule.

# --- Backend -----------------------------------------------------------------
# Select either the local AerSimulator or a real IBM Quantum computer.

if IBM_BACKEND is not None:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        print("Error: IBM_BACKEND is set but qiskit_ibm_runtime is not installed.")
        print("  Run: pip install qiskit-ibm-runtime")
        sys.exit(1)
    if IBM_TOKEN is not None:
        service = QiskitRuntimeService(token=IBM_TOKEN)
    else:
        service = QiskitRuntimeService()
    backend = service.backend(IBM_BACKEND)
    print(f"  Using real IBM Quantum hardware: {IBM_BACKEND}")
else:
    # max_parallel_experiments=0: use all available CPU cores when a batch of circuits
    # is submitted in a single backend.run() call. Gracefully degrades to sequential
    # execution if only one core is available. No effect in statevector mode.
    if SIM_SEED is not None:
        backend = AerSimulator(method=SIMULATOR_METHOD, seed_simulator=SIM_SEED,
                               max_parallel_experiments=0)
    else:
        backend = AerSimulator(method=SIMULATOR_METHOD, max_parallel_experiments=0)


# --- Preprocessing -----------------------------------------------------------

def preprocess_event(x_raw):
    """
    Pad the 12-dimensional feature vector to length 16 with zeros, then
    normalise to unit length to form a valid quantum-state amplitude vector.
    """
    x_padded = np.concatenate([x_raw, np.zeros(N_DIM - N_FEAT)])
    norm = np.linalg.norm(x_padded)
    if norm < 1e-12:
        raise ValueError("Feature vector has zero norm and cannot be normalised.")
    return x_padded / norm


# --- Amplitude embedding and circuit construction ----------------------------

def embed_state_in_qiskit(amplitudes):
    """Return a QuantumCircuit that prepares the given amplitude vector from |0000>."""
    qc = QuantumCircuit(N_QUBITS)
    qc.append(StatePreparation(amplitudes.tolist(), normalize=False), range(N_QUBITS))
    return qc


def build_event_circuit(amplitudes, theta_params):
    """
    Build and transpile the full circuit for one event: embedding + parameterised
    variational form. Transpiled once and reused across all passes by binding theta values
    via assign_parameters(), avoiding repeated transpilation overhead.
    """
    emb_qc  = embed_state_in_qiskit(amplitudes)
    full_qc = variational_form_fn(emb_qc, list(theta_params), n_layers)
    if not USE_STATEVECTOR:
        full_qc.measure_all()
    return transpile(full_qc, backend)


# --- Circuit measurement -----------------------------------------------------

def run_circuit_and_measure(transpiled_qc, theta_params, theta):
    """
    Bind theta to a pre-transpiled circuit and return P(qubit_0 = |1>).
    USE_STATEVECTOR=True  : exact result via statevector, no shot noise.
    USE_STATEVECTOR=False : estimate from Nm measurement shots.
    """
    bound = transpiled_qc.assign_parameters(
        {theta_params[i]: float(v) for i, v in enumerate(theta)}
    )
    if USE_STATEVECTOR:
        sv = Statevector.from_instruction(bound)
        return float(sv.probabilities([0])[1])
    counts = backend.run(bound, shots=Nm).result().get_counts()
    total  = sum(counts.values())
    return float(sum(v for k, v in counts.items() if k[-1] == '1') / total)


def _run_circuits_batch(batch_circuits, theta):
    """
    Evaluate all circuits in batch_circuits at the given theta and return
    an np.array of P(qubit_0 = |1>) values, one per event.

    Shot-based: submits ALL circuits in a single backend.run() call so AerSimulator
    can parallelise them across CPU cores (max_parallel_experiments=0). This reduces
    Python-to-Aer overhead from batch_size calls down to 1 call per theta value.
    Statevector: evaluates each circuit independently (no backend involved, same cost).
    """
    bound = [
        qc.assign_parameters({params[i]: float(v) for i, v in enumerate(theta)})
        for qc, params in batch_circuits
    ]
    if USE_STATEVECTOR:
        return np.array([
            float(Statevector.from_instruction(b).probabilities([0])[1])
            for b in bound
        ])
    job    = backend.run(bound, shots=Nm)
    result = job.result()
    values = []
    for i in range(len(bound)):
        counts = result.get_counts(i)
        total  = sum(counts.values())
        values.append(float(sum(v for k, v in counts.items() if k[-1] == '1') / total))
    return np.array(values)


# --- Loss function -----------------------------------------------------------

def compute_loss(p_values, event_labels, event_weights):
    """
    Weighted binary cross-entropy over a set of events:
      L_K = (1/N) * sum_i { w_i * [-z_i*ln(y_i) - (1-z_i)*ln(1-y_i)] }
    N is the number of events passed in (batch size or full dataset size).
    Uses sklearn log_loss for the weighted sum, then divides by N.
    """
    N      = len(p_values)
    p_clip = np.clip(p_values, 1e-7, 1 - 1e-7)
    raw    = log_loss(event_labels, p_clip, normalize=False, sample_weight=event_weights, labels=[0, 1])
    return float(raw / N)


# --- Shift rule classification ------------------------------------------------

def classify_shift_rules(qc_template, theta_params):
    """
    Inspect a transpiled parameterised circuit and classify each theta parameter
    by which shift rule its gate requires. Called once before training.
    Returns a list of length N_PARAMS: 'two_term' or 'four_term'.
    Raises ValueError for any parameter feeding an unrecognised gate type,
    so there is no silent failure or incorrect gradient.

    two_term  : gates exp(-iθ/2·P), P a Pauli, eigenvalues {+1,-1}.
                Shift π/2 gives the exact gradient.
    four_term : controlled-Pauli-rotation gates, generator eigenvalues {-1,0,+1}.
                Four-term rule (shifts π/2 and π) gives the exact gradient.
    """
    TWO_TERM_GATES  = {'rx', 'ry', 'rz'}
    FOUR_TERM_GATES = {'crx', 'cry', 'crz'}

    param_to_rule = {}
    for instr in qc_template.data:
        gate_name = instr.operation.name
        for p in instr.operation.params:
            for param in getattr(p, 'parameters', []):
                if gate_name in TWO_TERM_GATES:
                    param_to_rule[param] = 'two_term'
                elif gate_name in FOUR_TERM_GATES:
                    param_to_rule[param] = 'four_term'
                else:
                    raise ValueError(
                        f"Parameter '{param.name}' feeds gate '{gate_name}', which has "
                        f"no known parameter-shift rule. Add its rule to "
                        f"classify_shift_rules before using it."
                    )
    return [param_to_rule[theta_params[i]] for i in range(len(theta_params))]


# --- Batch gradient via parameter shift rule ---------------------------------

def compute_batch_gradient_and_loss(batch_circuits, batch_labels, batch_weights, theta,
                                    shift_rules):
    """
    Compute the gradient of the batch loss w.r.t. theta and the batch loss value.

    Steps:
      1. Forward pass: run all batch circuits at current theta -> p_current[j].
         The batch loss is computed here for free (no extra circuit runs needed).
      2. Per-event analytical gradient of loss w.r.t. P(|1>):
           dL_j/dp_j = w_j * (-z_j / p_j  +  (1-z_j) / (1-p_j))
      3. Per theta component k the correct exact shift rule is applied:
           two_term  : grad[k] = (1/N) * dot(dL_dp, (p+ - p-) / 2)
                       p± evaluated at theta ± π/2.
           four_term : grad[k] = (1/N) * dot(dL_dp, d1*(fa+−fa−) − d2*(fb+−fb−))
                       a-shifts ±π/2, b-shifts ±π, d1=1/2, d2=(√2−1)/4.
      Total circuit evaluations: (1 + 2*n_two + 4*n_four) * batch_size.

    Returns (gradient_array, batch_loss_scalar).
    batch_loss is the pre-step loss (computed before the Adam update).
    """
    N  = len(batch_circuits)
    D1 = 0.5
    D2 = (np.sqrt(2) - 1) / 4

    # Forward pass: all batch events in one backend.run() call.
    p_current  = _run_circuits_batch(batch_circuits, theta)
    batch_loss = compute_loss(p_current, batch_labels, batch_weights)

    # Analytical per-event gradient of loss w.r.t. P(|1>).
    p_clip = np.clip(p_current, 1e-7, 1 - 1e-7)
    dL_dp  = batch_weights * (
        -batch_labels / p_clip + (1 - batch_labels) / (1 - p_clip)
    )

    gradients = np.zeros(len(theta))
    for k in range(len(theta)):
        if shift_rules[k] == 'two_term':
            tp    = theta.copy(); tp[k] += np.pi / 2
            tm    = theta.copy(); tm[k] -= np.pi / 2
            p_plus   = _run_circuits_batch(batch_circuits, tp)
            p_minus  = _run_circuits_batch(batch_circuits, tm)
            dpdtheta = (p_plus - p_minus) / 2.0
        else:  # four_term
            ta_p = theta.copy(); ta_p[k] += np.pi / 2
            ta_m = theta.copy(); ta_m[k] -= np.pi / 2
            tb_p = theta.copy(); tb_p[k] += np.pi
            tb_m = theta.copy(); tb_m[k] -= np.pi
            f_ap = _run_circuits_batch(batch_circuits, ta_p)
            f_am = _run_circuits_batch(batch_circuits, ta_m)
            f_bp = _run_circuits_batch(batch_circuits, tb_p)
            f_bm = _run_circuits_batch(batch_circuits, tb_m)
            dpdtheta = D1 * (f_ap - f_am) - D2 * (f_bp - f_bm)

        gradients[k] = float(np.dot(dL_dp, dpdtheta) / N)

    return gradients, batch_loss


# --- Global evaluation (all events) ------------------------------------------

def evaluate_global_loss(circuit_list, event_labels, event_weights, theta, desc="Eval"):
    """
    Run all events at the given theta and return the global loss.
    Processes events in batch_size-sized chunks to limit memory usage while
    still submitting each chunk as a single backend.run() call for efficiency.
    """
    n_total  = len(circuit_list)
    p_values = np.empty(n_total)
    n_chunks = (n_total + batch_size - 1) // batch_size
    for c in tqdm(range(n_chunks), desc=f"  {desc}", unit="batch", leave=False):
        s = c * batch_size
        e = min(s + batch_size, n_total)
        p_values[s:e] = _run_circuits_batch(circuit_list[s:e], theta)
    return compute_loss(p_values, event_labels, event_weights)


# --- Output utilities --------------------------------------------------------

def record_final_circuit(theta_final):
    """Run the variational form once with the trained theta to record the gate sequence."""
    qc = QuantumCircuit(N_QUBITS)
    variational_form_fn(qc, np.array(theta_final), n_layers)
    gates = []
    for instr in qc.data:
        gates.append({
            'name'  : instr.operation.name,
            'qubits': [qc.find_bit(q).index for q in instr.qubits],
            'params': [float(p) for p in instr.operation.params],
        })
    return gates


def generate_theta_labels(gate_sequence):
    """Return a human-readable label for each parameterised gate in the variational form."""
    out = []
    for gate in gate_sequence:
        if gate['params']:
            name   = gate['name']
            qubits = gate['qubits']
            val    = gate['params'][0]
            if name in ('crz', 'crx', 'cry'):
                out.append(
                    f"CTRL-{name[1:].upper()}(ctrl=q{qubits[0]}, tgt=q{qubits[1]}) = {val:.6f} rad"
                )
            else:
                out.append(f"{name.upper()}(q{qubits[0]}) = {val:.6f} rad")
    return out


# =============================================================================
# TRAINING SETUP
# =============================================================================
# Pre-builds and transpiles one full circuit per event (embedding + variational
# form), initialises theta, and creates the sklearn Adam optimiser.

# Pre-build and transpile one full circuit per event (embedding + variational form).
# All N_iter passes reuse these transpiled circuits by binding parameter values.
print(f"Pre-building and transpiling circuits ({N_EVENTS} events)...")
all_circuits = []
for x_raw in tqdm(events, desc="Building", unit="event", leave=True):
    amplitudes   = preprocess_event(x_raw)
    theta_params = ParameterVector('theta', N_PARAMS)
    all_circuits.append((build_event_circuit(amplitudes, theta_params), theta_params))
print()

# Classify each theta parameter by the shift rule its gate requires.
# Uses the first circuit as template — all circuits share the same variational form.
shift_rules = classify_shift_rules(all_circuits[0][0], all_circuits[0][1])
_n_two  = shift_rules.count('two_term')
_n_four = shift_rules.count('four_term')
print(f"  Shift rule classification: {_n_two} two-term (Rx/Ry/Rz), "
      f"{_n_four} four-term (CRx/CRy/CRz)\n")

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

print(f"Starting training: {n_iter} passes x {N_BATCHES_PER_ITER} batches "
      f"x {batch_size} events/batch")
print(f"  Simulation : {'statevector (exact)' if USE_STATEVECTOR else f'Nm={Nm} shots'}")
print(f"  Adam       : sklearn AdamOptimizer, Kingma & Ba 2014  "
      f"alpha={ALPHA}  beta1={BETA1}  beta2={BETA2}\n")


# =============================================================================
# PRE-TRAINING EVALUATIONS  ("start" checkpoints, before any Adam step)
# =============================================================================

if global_do_start or batch_do_start:
    _b_start = "--------"
    _g_start = "--------"
    if global_do_start:
        loss = evaluate_global_loss(
            all_circuits, labels, weights, theta_current, desc="  Global [start]"
        )
        global_eval_results["start"] = float(loss)
        _g_start = f"{loss:.6f}"
    if batch_do_start:
        # Evaluate the first batch with the initial random theta before any Adam step.
        p_start    = _run_circuits_batch(all_circuits[:batch_size], theta_current)
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

    # Optionally reshuffle the event order at the start of each pass after the first.
    if RESHUFFLE_BETWEEN_PASSES and current_iter > 1:
        if PASS_RESHUFFLE_SEED is not None:
            rng_reshuffle = np.random.default_rng(seed=PASS_RESHUFFLE_SEED + current_iter)
        else:
            rng_reshuffle = np.random.default_rng()
        perm          = rng_reshuffle.permutation(N_EVENTS)
        pass_circuits = [all_circuits[i] for i in perm]
        pass_labels   = labels[perm]
        pass_weights  = weights[perm]
    else:
        pass_circuits = all_circuits
        pass_labels   = labels
        pass_weights  = weights

    # Sweep through all batches for this pass, with a per-batch progress bar.
    with tqdm(range(N_BATCHES_PER_ITER),
              desc=f"  Pass {current_iter:{n_width}d}/{n_iter}",
              unit="batch", leave=False, position=1) as batch_bar:
        for batch_idx in batch_bar:
            start = batch_idx * batch_size
            end   = start + batch_size

            b_circuits = pass_circuits[start:end]
            b_labels   = pass_labels[start:end]
            b_weights  = pass_weights[start:end]

            # Compute gradient and pre-step batch loss via parameter shift rule.
            grad, current_batch_loss = compute_batch_gradient_and_loss(
                b_circuits, b_labels, b_weights, theta_current, shift_rules
            )

            # One Adam step: sklearn AdamOptimizer updates theta_current in place.
            # Adam step.
            optimizer.update_params([theta_current], [grad])

            # Track the last batch loss of this pass for BATCH_EVAL_CHECKPOINTS.
            last_batch_loss = current_batch_loss

    pass_bar.update(1)

    # Eval checkpoints: batch (last batch pre-step loss) and/or global (all events).
    # Both are shown on the same line for easy comparison.
    if current_iter in batch_ckpt_set or current_iter in global_ckpt_set:
        _b_str = "--------"
        _g_str = "--------"
        if current_iter in batch_ckpt_set:
            batch_eval_results[str(current_iter)] = float(last_batch_loss)
            _b_str = f"{last_batch_loss:.6f}"
        if current_iter in global_ckpt_set:
            g_loss = evaluate_global_loss(
                all_circuits, labels, weights, theta_current,
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

# --- JSON results file -------------------------------------------------------

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
        "USE_STATEVECTOR"         : USE_STATEVECTOR,
        "SIMULATOR_METHOD"        : SIMULATOR_METHOD,
        "Nm"                      : Nm,
        "SIM_SEED"                : SIM_SEED,
        "IBM_BACKEND"             : IBM_BACKEND,
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


# --- Standalone classifier script --------------------------------------------

def gate_to_qiskit_line(gate):
    """Convert a recorded gate dict to one line of Qiskit Python code."""
    name   = gate['name']
    qubits = gate['qubits']
    params = gate['params']
    if name == 'cx':
        return f"    qc.cx({qubits[0]}, {qubits[1]})"
    elif name == 'cz':
        return f"    qc.cz({qubits[0]}, {qubits[1]})"
    elif name == 'h':
        return f"    qc.h({qubits[0]})"
    elif name in ('crz', 'crx', 'cry') and params:
        return f"    qc.{name}({params[0]:.10f}, {qubits[0]}, {qubits[1]})"
    elif params:
        return f"    qc.{name}({params[0]:.10f}, {qubits[0]})"
    else:
        return f"    # Unrecognised gate {name} on {qubits}"


gate_lines     = "\n".join(gate_to_qiskit_line(g) for g in gate_sequence)
theta_literal  = repr(theta_final)
labels_literal = repr(theta_labels)

classifier_script = textwrap.dedent(f"""\
    # classifier.py
    # Standalone classifier — generated by PQC_HEP_training_file_Qiskit.py
    # Run timestamp  : {timestamp}
    # Circuit        : [{circuit_number}] {circuit_description}
    # Layers         : {n_layers}  |  Parameters: {N_PARAMS}
    # Trained on     : {sig_name_only} + {bkg_name_only}  ({N_EVENTS} events used)
    #
    # To classify a new event: set your_event_features and run this script.
    # Requires: qiskit, qiskit-aer, numpy

    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import StatePreparation
    from qiskit_aer import AerSimulator

    N_QUBITS    = {N_QUBITS}
    N_FEAT      = {N_FEAT}
    N_DIM       = {N_DIM}
    Nm_classify = {Nm}

    THETA_FINAL  = {theta_literal}
    THETA_LABELS = {labels_literal}

    your_event_features = np.array([0.0] * N_FEAT)   # <-- replace with event data

    def preprocess(x_raw):
        x_padded = np.concatenate([x_raw, np.zeros(N_DIM - N_FEAT)])
        norm = np.linalg.norm(x_padded)
        if norm < 1e-12:
            raise ValueError("Zero-norm feature vector.")
        return x_padded / norm

    amplitudes = preprocess(your_event_features)
    qc = QuantumCircuit(N_QUBITS)
    qc.append(StatePreparation(amplitudes.tolist(), normalize=False), range(N_QUBITS))

    theta_final  = THETA_FINAL
    theta_labels = THETA_LABELS
{gate_lines}
    qc.measure_all()

    backend = AerSimulator()
    counts  = backend.run(transpile(qc, backend), shots=Nm_classify).result().get_counts()
    total   = sum(counts.values())
    p1      = sum(v for k, v in counts.items() if k[-1] == '1') / total
    y_pred  = float(p1)

    print(f"Classification probability = {{y_pred:.4f}}")
    print(f"Classification result      = {{'SIGNAL' if y_pred >= 0.5 else 'BACKGROUND'}}")
""")

classifier_path = os.path.join(output_dir, "classifier.py")
with open(classifier_path, "w") as f:
    f.write(classifier_script)
print(f"  [2/2] Classifier   : {classifier_path}")

# --- Final summary -----------------------------------------------------------

print(f"\nTraining complete.")
print(f"  Circuit      : [{circuit_number}] {circuit_description}")
print(f"  Layers       : {n_layers}  |  Parameters: {N_PARAMS}")
print(f"  Events       : {N_EVENTS} used of {len(all_feats)} loaded")
print(f"  N_iter       : {n_iter}  |  Batches/iter: {N_BATCHES_PER_ITER}  |  Batch size: {batch_size}")
print(f"  theta_final  : {np.round(theta_final, 4)}")
if global_eval_results:
    last_g = sorted((k for k in global_eval_results if k != "start"), key=int)
    if last_g:
        print(f"  Last global eval (n={last_g[-1]}): loss={global_eval_results[last_g[-1]]:.6f}")
if batch_eval_results:
    last_b = sorted((k for k in batch_eval_results if k != "start"), key=int)
    if last_b:
        print(f"  Last batch  eval (n={last_b[-1]}): loss={batch_eval_results[last_b[-1]]:.6f}")
print(f"  Output folder: {output_dir}/")

if SHOW_ELAPSED_TIME:
    _elapsed  = time.time() - _t_start
    _sim_info = 'statevector (exact)' if USE_STATEVECTOR else f'Nm={Nm} shots'
    print(f"\n  Elapsed time : {_elapsed:.1f} s  ({_elapsed/60:.1f} min)")
    print(f"  Per event    : {_elapsed/N_EVENTS:.3f} s/event  "
          f"(N_iter={n_iter}, {_sim_info}, {N_EVENTS} events)")
