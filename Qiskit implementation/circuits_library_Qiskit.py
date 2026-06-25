"""
circuits_library_Qiskit.py

Author: Adriano Pinto Claro da Fonseca
Email: apcf@topfonseca.com
Institutional email: adriano.fonseca@tecnico.ulisboa.pt
Affiliation: Student at Instituto Superior Técnico, Universidade de Lisboa

Description:
    Defines 19 parameterized ansatz circuits for use as W(theta) in the PQC pipeline.
    All circuits are implemented natively in Qiskit — no PyQuil dependency.
    Circuit templates follow Figure 2 of Sim et al. (2019) — see references in README.md.

Each circuit function has the following signature:
    circuit_N(qc, theta, n_layers) -> tuple[QuantumCircuit, list[str]]

    qc       : Qiskit QuantumCircuit containing the already-embedded quantum state.
    theta    : 1-D array (or ParameterVector) holding all rotation angles for all
               layers. Accepts both plain floats and Qiskit Parameter objects so
               the same function works for numeric evaluation and for building a
               parameterized circuit that can be transpiled once and reused.
    n_layers : number of times the repeatable layer block is applied.

Returns:
    qc         : the same QuantumCircuit object passed in (modified in-place).
    parameterized_gate_names : list of strings, one per variational parameter, naming the gate
                 that consumes that parameter (e.g. 'rx', 'rz', 'crz'). The list
                 is in parameter order, so parameterized_gate_names[k] is the gate for theta[k].
                 Non-parameterized gates (cx, cz, h) are not included.

Parameter layout within theta:
    Indices 0 .. fixed_params-1                          : fixed part, applied once.
    Indices fixed_params .. fixed_params+layer_params-1  : layer 1.
    Indices fixed_params+layer_params .. -1              : layer 2, 3, ... etc.

Total parameters = fixed_params + n_layers * layer_params.

For most circuits fixed_params = 0 and the entire circuit is the repeatable layer.
Circuit 10 is the exception: it has an initial RY block outside the repeatable layer.

CIRCUITS registry format per entry:
    (function, fixed_params, layer_params, description)

Index 0 is None so that circuit_number on the command line maps directly to the
circuit label in the source paper (circuit_number=1 selects CIRCUITS[1], etc.).
"""

# Implementation note on parameter indexing:
# Each circuit function uses a local integer counter called current_parameter
# that starts at 0 and increments by 1 each time a rotation gate consumes a
# parameter from theta. This means theta[0] always goes to the first rotation
# gate of the first layer, theta[1] to the second, and so on continuously
# across all layers. Exception: Circuit 10 has a 4-parameter pre-layer block
# (fixed RY gates applied before all layer repetitions), so theta[0]–theta[3]
# go to that block and theta[4] starts the first layer. The total number of
# parameters consumed equals fixed_params + n_layers * layer_params, which
# must match N_PARAMS in the training file. If you add a new circuit, count
# the rotation gates carefully.

# Gate naming used throughout this library:
#   qc.rx(theta, q)              : single-qubit X rotation by angle theta on qubit q
#   qc.ry(theta, q)              : single-qubit Y rotation by angle theta on qubit q
#   qc.rz(theta, q)              : single-qubit Z rotation by angle theta on qubit q
#   qc.cx(ctrl, tgt)             : CNOT gate — ctrl is control qubit, tgt is target (no parameter)
#   qc.cz(q1, q2)                : controlled-Z gate between q1 and q2 (no parameter)
#   qc.h(q)                      : Hadamard gate on qubit q (no parameter)
#   qc.crz(theta, ctrl, target)  : controlled-RZ — ctrl is control, target is target
#   qc.crx(theta, ctrl, target)  : controlled-RX — ctrl is control, target is target
#
# Non-parameterized gates (cx, cz, h) are NOT included in the returned parameterized_gate_names list.
# All gate methods modify qc in-place; the returned qc is the same object as the input.


# Circuit 1
# One repeatable layer: RX then RZ on each qubit, no entanglement.
# Parameters per layer: 8 (2 per qubit × 4 qubits).

DESC_1  = "Circuit 1: RX-RZ on each qubit, no entanglement. 8 parameters per layer."
PRE_LAYER_PARAMETERS_1 = 0
PARAMETERS_PER_LAYER_1 = 8

def circuit_1(qc, theta, n_layers):
    current_parameter = 0
    parameterized_gate_names = []

    for _ in range(n_layers):

        for i in range(4):
            qc.rx(theta[current_parameter], i)
            parameterized_gate_names.append('rx')
            current_parameter += 1

            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

    return qc, parameterized_gate_names



# Circuit 2
# One repeatable layer: RX then RZ on each qubit, followed by a CNOT ladder
# descending from qubit 3 down to qubit 0 (CX(3,2), CX(2,1), CX(1,0)).
# Parameters per layer: 8 (2 per qubit × 4 qubits). No parameters on CX gates.

DESC_2  = "Circuit 2: RX-RZ on each qubit, CNOT ladder descending q3->q2->q1->q0. 8 parameters per layer."
PRE_LAYER_PARAMETERS_2 = 0
PARAMETERS_PER_LAYER_2 = 8

def circuit_2(qc, theta, n_layers):
    current_parameter = 0
    parameterized_gate_names = []

    for _ in range(n_layers):

        for i in range(4):
            qc.rx(theta[current_parameter], i)
            parameterized_gate_names.append('rx')
            current_parameter += 1

            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

        for i in range(3, 0, -1):
            qc.cx(i, i-1)

    return qc, parameterized_gate_names



# Circuit 3
# One repeatable layer: RX then RZ on each qubit, followed by a controlled-RZ
# ladder descending from qubit 3 to qubit 0. Each controlled-RZ uses the
# higher-index qubit as control and the adjacent lower-index qubit as target:
# CRZ(q3->q2), CRZ(q2->q1), CRZ(q1->q0).
# Parameters per layer: 11 (8 from RX-RZ block + 3 from CRZ gates).

DESC_3  = "Circuit 3: RX-RZ on each qubit, controlled-RZ ladder descending q3->q2->q1->q0. 11 parameters per layer."
PRE_LAYER_PARAMETERS_3 = 0
PARAMETERS_PER_LAYER_3 = 11

def circuit_3(qc, theta, n_layers):
    current_parameter = 0
    parameterized_gate_names = []

    for _ in range(n_layers):

        for i in range(4):
            qc.rx(theta[current_parameter], i)
            parameterized_gate_names.append('rx')
            current_parameter += 1

            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

        for i in range(3, 0, -1):
            qc.crz(theta[current_parameter], i, i-1)
            parameterized_gate_names.append('crz')
            current_parameter += 1

    return qc, parameterized_gate_names



# Circuit 4
# One repeatable layer: RX then RZ on each qubit, followed by a controlled-RX
# ladder descending from qubit 3 to qubit 0. Identical structure to Circuit 3
# with CRX replacing CRZ:
# CRX(q3->q2), CRX(q2->q1), CRX(q1->q0).
# Parameters per layer: 11 (8 from RX-RZ block + 3 from CRX gates).

DESC_4  = "Circuit 4: RX-RZ on each qubit, controlled-RX ladder descending q3->q2->q1->q0. 11 parameters per layer."
PRE_LAYER_PARAMETERS_4 = 0
PARAMETERS_PER_LAYER_4 = 11

def circuit_4(qc, theta, n_layers):
    current_parameter = 0
    parameterized_gate_names = []

    for _ in range(n_layers):

        for i in range(4):
            qc.rx(theta[current_parameter], i)
            parameterized_gate_names.append('rx')
            current_parameter += 1

            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

        for i in range(3, 0, -1):
            qc.crx(theta[current_parameter], i, i-1)
            parameterized_gate_names.append('crx')
            current_parameter += 1

    return qc, parameterized_gate_names



# Circuit 5
# One repeatable layer structured in three blocks:
#   Block 1: RX then RZ on all qubits (8 parameters).
#   Block 2: full entanglement via CRZ gates. Each of the 4 qubits
#            acts as control once, targeting all 3 remaining qubits:
#            q3 controls q2, q1, q0 (3 params);
#            q2 controls q3, q1, q0 (3 params);
#            q1 controls q3, q2, q0 (3 params);
#            q0 controls q3, q2, q1 (3 params). Total: 12 parameters.
#   Block 3: RX then RZ on all qubits (8 parameters).
# Parameters per layer: 28 (8 + 12 + 8).

DESC_5  = "Circuit 5: RX-RZ on each qubit, full controlled-RZ entanglement block (each qubit controls all others), final RX-RZ. 28 parameters per layer."
PRE_LAYER_PARAMETERS_5 = 0
PARAMETERS_PER_LAYER_5 = 28

def circuit_5(qc, theta, n_layers):
    current_parameter = 0
    parameterized_gate_names = []

    for _ in range(n_layers):

        for i in range(4):
            qc.rx(theta[current_parameter], i)
            parameterized_gate_names.append('rx')
            current_parameter += 1

            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

        for control_q in range(3, -1, -1):
            for target_q in range(3, -1, -1):
                if control_q != target_q:

                    qc.crz(theta[current_parameter], control_q, target_q)
                    parameterized_gate_names.append('crz')
                    current_parameter += 1

        for i in range(4):
            qc.rx(theta[current_parameter], i)
            parameterized_gate_names.append('rx')
            current_parameter += 1

            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

    return qc, parameterized_gate_names



# Circuit 6
# One repeatable layer structured in three blocks:
#   Block 1: RX then RZ on all qubits (8 parameters).
#   Block 2: full entanglement via CRX gates. Each of the 4 qubits
#            acts as control once, targeting all 3 remaining qubits:
#            q3 controls q2, q1, q0 (3 params);
#            q2 controls q3, q1, q0 (3 params);
#            q1 controls q3, q2, q0 (3 params);
#            q0 controls q3, q2, q1 (3 params). Total: 12 parameters.
#   Block 3: RX then RZ on all qubits (8 parameters).
# Identical structure to Circuit 5 with CRX replacing CRZ.
# Parameters per layer: 28 (8 + 12 + 8).

DESC_6  = "Circuit 6: RX-RZ on each qubit, full controlled-RX entanglement block (each qubit controls all others), final RX-RZ. 28 parameters per layer."
PRE_LAYER_PARAMETERS_6 = 0
PARAMETERS_PER_LAYER_6 = 28

def circuit_6(qc, theta, n_layers):
    current_parameter = 0
    parameterized_gate_names = []

    for _ in range(n_layers):

        for i in range(4):
            qc.rx(theta[current_parameter], i)
            parameterized_gate_names.append('rx')
            current_parameter += 1

            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

        for control_q in range(3, -1, -1):
            for target_q in range(3, -1, -1):
                if control_q != target_q:

                    qc.crx(theta[current_parameter], control_q, target_q)
                    parameterized_gate_names.append('crx')
                    current_parameter += 1

        for i in range(4):
            qc.rx(theta[current_parameter], i)
            parameterized_gate_names.append('rx')
            current_parameter += 1

            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

    return qc, parameterized_gate_names



# Circuit 7
# One repeatable layer structured as follows:
#   Block 1: RX then RZ on all qubits (8 parameters).
#   Block 2: two CRZ gates in parallel pairs:
#            CRZ(q1->q0) and CRZ(q3->q2) (2 parameters).
#   Block 3: RX then RZ on all qubits (8 parameters).
#   Block 4: one CRZ gate connecting the two pairs:
#            CRZ(q2->q1) (1 parameter).
# Parameters per layer: 19 (8 + 2 + 8 + 1).

DESC_7  = "Circuit 7: RX-RZ block, parallel controlled-RZ pairs (q1->q0, q3->q2), RX-RZ block, controlled-RZ (q2->q1). 19 parameters per layer."
PRE_LAYER_PARAMETERS_7 = 0
PARAMETERS_PER_LAYER_7 = 19

def circuit_7(qc, theta, n_layers):
    current_parameter = 0
    parameterized_gate_names = []

    for _ in range(n_layers):

        for i in range(4):
            qc.rx(theta[current_parameter], i)
            parameterized_gate_names.append('rx')
            current_parameter += 1

            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

        qc.crz(theta[current_parameter], 1, 0)
        parameterized_gate_names.append('crz')
        current_parameter += 1

        qc.crz(theta[current_parameter], 3, 2)
        parameterized_gate_names.append('crz')
        current_parameter += 1

        for i in range(4):
            qc.rx(theta[current_parameter], i)
            parameterized_gate_names.append('rx')
            current_parameter += 1

            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

        qc.crz(theta[current_parameter], 2, 1)
        parameterized_gate_names.append('crz')
        current_parameter += 1

    return qc, parameterized_gate_names



# Circuit 8
# One repeatable layer structured as follows:
#   Block 1: RX then RZ on all qubits (8 parameters).
#   Block 2: two CRX gates in parallel pairs:
#            CRX(q1->q0) and CRX(q3->q2) (2 parameters).
#   Block 3: RX then RZ on all qubits (8 parameters).
#   Block 4: one CRX gate connecting the two pairs:
#            CRX(q2->q1) (1 parameter).
# Identical structure to Circuit 7 with CRX replacing CRZ.
# Parameters per layer: 19 (8 + 2 + 8 + 1).

DESC_8  = "Circuit 8: RX-RZ block, parallel controlled-RX pairs (q1->q0, q3->q2), RX-RZ block, controlled-RX (q2->q1). 19 parameters per layer."
PRE_LAYER_PARAMETERS_8 = 0
PARAMETERS_PER_LAYER_8 = 19

def circuit_8(qc, theta, n_layers):
    current_parameter = 0
    parameterized_gate_names = []

    for _ in range(n_layers):

        for i in range(4):
            qc.rx(theta[current_parameter], i)
            parameterized_gate_names.append('rx')
            current_parameter += 1

            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

        qc.crx(theta[current_parameter], 1, 0)
        parameterized_gate_names.append('crx')
        current_parameter += 1

        qc.crx(theta[current_parameter], 3, 2)
        parameterized_gate_names.append('crx')
        current_parameter += 1

        for i in range(4):
            qc.rx(theta[current_parameter], i)
            parameterized_gate_names.append('rx')
            current_parameter += 1

            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

        qc.crx(theta[current_parameter], 2, 1)
        parameterized_gate_names.append('crx')
        current_parameter += 1

    return qc, parameterized_gate_names



# Circuit 9
# One repeatable layer structured as follows:
#   Block 1: H on each qubit (no parameters).
#   Block 2: CZ ladder descending: CZ(q3,q2), CZ(q2,q1), CZ(q1,q0) (no parameters).
#   Block 3: RX on each qubit (4 parameters).
# Parameters per layer: 4 (one RX per qubit).

DESC_9  = "Circuit 9: H on each qubit, CZ ladder descending q3->q2->q1->q0, RX on each qubit. 4 parameters per layer."
PRE_LAYER_PARAMETERS_9 = 0
PARAMETERS_PER_LAYER_9 = 4

def circuit_9(qc, theta, n_layers):
    current_parameter = 0
    parameterized_gate_names = []

    for _ in range(n_layers):

        for i in range(4):
            qc.h(i)

        for i in range(3, 0, -1):
            qc.cz(i, i-1)

        for i in range(4):
            qc.rx(theta[current_parameter], i)
            parameterized_gate_names.append('rx')
            current_parameter += 1

    return qc, parameterized_gate_names



# Circuit 10
# Fixed part (applied once, outside the repeatable layer):
#   RY on each qubit (4 parameters).
# Repeatable layer:
#   Block 1: CZ gates forming a descending ladder plus one additional long-range
#            connection: CZ(q3,q2), CZ(q2,q1), CZ(q1,q0), CZ(q3,q0) (no parameters).
#   Block 2: RY on each qubit (4 parameters).
# Fixed parameters: 4. Parameters per layer: 4.

DESC_10  = "Circuit 10: fixed RY on each qubit, then per layer [CZ ladder descending q3->q2->q1->q0 plus CZ(q3,q0), RY on each qubit]. 4 fixed parameters, 4 parameters per layer."
PRE_LAYER_PARAMETERS_10 = 4
PARAMETERS_PER_LAYER_10 = 4

def circuit_10(qc, theta, n_layers):
    current_parameter = 0
    parameterized_gate_names = []

    for i in range(4):
        qc.ry(theta[current_parameter], i)
        parameterized_gate_names.append('ry')
        current_parameter += 1

    for _ in range(n_layers):

        for i in range(3, 0, -1):
            qc.cz(i, i-1)

        qc.cz(3, 0)

        for i in range(4):
            qc.ry(theta[current_parameter], i)
            parameterized_gate_names.append('ry')
            current_parameter += 1

    return qc, parameterized_gate_names



# Circuit 11
# One repeatable layer structured as follows:
#   Block 1: RY then RZ on all qubits (8 parameters).
#   Block 2: two parallel CNOT gates: CX(q1,q0) and CX(q3,q2) (no parameters).
#   Block 3: RY then RZ on qubits 1 and 2 only (4 parameters).
#   Block 4: one CNOT gate connecting the two pairs: CX(q2,q1) (no parameters).
# Parameters per layer: 12 (8 + 4).

DESC_11  = "Circuit 11: RY-RZ on all qubits, parallel CNOT(q1->q0) and CNOT(q3->q2), RY-RZ on q1 and q2, CNOT(q2->q1). 12 parameters per layer."
PRE_LAYER_PARAMETERS_11 = 0
PARAMETERS_PER_LAYER_11 = 12

def circuit_11(qc, theta, n_layers):
    current_parameter = 0
    parameterized_gate_names = []

    for _ in range(n_layers):

        for i in range(4):
            qc.ry(theta[current_parameter], i)
            parameterized_gate_names.append('ry')
            current_parameter += 1

            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

        qc.cx(1, 0)

        qc.cx(3, 2)

        for i in range(1, 3):
            qc.ry(theta[current_parameter], i)
            parameterized_gate_names.append('ry')
            current_parameter += 1

            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

        qc.cx(2, 1)

    return qc, parameterized_gate_names



# Circuit 12
# One repeatable layer structured as follows:
#   Block 1: RY then RZ on all qubits (8 parameters).
#   Block 2: two parallel CZ gates: CZ(q1,q0) and CZ(q3,q2) (no parameters).
#   Block 3: RY then RZ on qubits 1 and 2 only (4 parameters).
#   Block 4: one CZ gate connecting the two pairs: CZ(q2,q1) (no parameters).
# Identical structure to Circuit 11 with CZ replacing CX.
# Parameters per layer: 12 (8 + 4).

DESC_12  = "Circuit 12: RY-RZ on all qubits, parallel CZ(q1,q0) and CZ(q3,q2), RY-RZ on q1 and q2, CZ(q2,q1). 12 parameters per layer."
PRE_LAYER_PARAMETERS_12 = 0
PARAMETERS_PER_LAYER_12 = 12

def circuit_12(qc, theta, n_layers):
    current_parameter = 0
    parameterized_gate_names = []

    for _ in range(n_layers):

        for i in range(4):
            qc.ry(theta[current_parameter], i)
            parameterized_gate_names.append('ry')
            current_parameter += 1

            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

        qc.cz(1, 0)

        qc.cz(3, 2)

        for i in range(1, 3):
            qc.ry(theta[current_parameter], i)
            parameterized_gate_names.append('ry')
            current_parameter += 1

            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

        qc.cz(2, 1)

    return qc, parameterized_gate_names



# Circuit 13
# One repeatable layer structured as follows:
#   Block 1: RY on all qubits (4 parameters).
#   Block 2: CRZ ring descending:
#            CRZ(q3->q0), CRZ(q2->q3), CRZ(q1->q2), CRZ(q0->q1) (4 parameters).
#   Block 3: RY on all qubits (4 parameters).
#   Block 4: CRZ ring ascending:
#            CRZ(q3->q2), CRZ(q0->q3), CRZ(q1->q0), CRZ(q2->q1) (4 parameters).
# Parameters per layer: 16 (4 + 4 + 4 + 4).

DESC_13  = "Circuit 13: RY on all qubits, controlled-RZ descending ring, RY on all qubits, controlled-RZ ascending ring. 16 parameters per layer."
PRE_LAYER_PARAMETERS_13 = 0
PARAMETERS_PER_LAYER_13 = 16

def circuit_13(qc, theta, n_layers):
    current_parameter = 0
    parameterized_gate_names = []

    for _ in range(n_layers):

        for i in range(4):
            qc.ry(theta[current_parameter], i)
            parameterized_gate_names.append('ry')
            current_parameter += 1

        qc.crz(theta[current_parameter], 3, 0)
        parameterized_gate_names.append('crz')
        current_parameter += 1

        for i in range(3, 0, -1):
            qc.crz(theta[current_parameter], i-1, i)
            parameterized_gate_names.append('crz')
            current_parameter += 1

        for i in range(4):
            qc.ry(theta[current_parameter], i)
            parameterized_gate_names.append('ry')
            current_parameter += 1

        qc.crz(theta[current_parameter], 3, 2)
        parameterized_gate_names.append('crz')
        current_parameter += 1

        qc.crz(theta[current_parameter], 0, 3)
        parameterized_gate_names.append('crz')
        current_parameter += 1

        for i in range(2):
            qc.crz(theta[current_parameter], i+1, i)
            parameterized_gate_names.append('crz')
            current_parameter += 1

    return qc, parameterized_gate_names



# Circuit 14
# One repeatable layer structured as follows:
#   Block 1: RY on all qubits (4 parameters).
#   Block 2: CRX ring descending:
#            CRX(q3->q0), CRX(q2->q3), CRX(q1->q2), CRX(q0->q1) (4 parameters).
#   Block 3: RY on all qubits (4 parameters).
#   Block 4: CRX ring ascending:
#            CRX(q3->q2), CRX(q0->q3), CRX(q1->q0), CRX(q2->q1) (4 parameters).
# Identical structure to Circuit 13 with CRX replacing CRZ.
# Parameters per layer: 16 (4 + 4 + 4 + 4).

DESC_14  = "Circuit 14: RY on all qubits, controlled-RX descending ring, RY on all qubits, controlled-RX ascending ring. 16 parameters per layer."
PRE_LAYER_PARAMETERS_14 = 0
PARAMETERS_PER_LAYER_14 = 16

def circuit_14(qc, theta, n_layers):
    current_parameter = 0
    parameterized_gate_names = []

    for _ in range(n_layers):

        for i in range(4):
            qc.ry(theta[current_parameter], i)
            parameterized_gate_names.append('ry')
            current_parameter += 1

        qc.crx(theta[current_parameter], 3, 0)
        parameterized_gate_names.append('crx')
        current_parameter += 1

        for i in range(3, 0, -1):
            qc.crx(theta[current_parameter], i-1, i)
            parameterized_gate_names.append('crx')
            current_parameter += 1

        for i in range(4):
            qc.ry(theta[current_parameter], i)
            parameterized_gate_names.append('ry')
            current_parameter += 1

        qc.crx(theta[current_parameter], 3, 2)
        parameterized_gate_names.append('crx')
        current_parameter += 1

        qc.crx(theta[current_parameter], 0, 3)
        parameterized_gate_names.append('crx')
        current_parameter += 1

        for i in range(2):
            qc.crx(theta[current_parameter], i+1, i)
            parameterized_gate_names.append('crx')
            current_parameter += 1

    return qc, parameterized_gate_names



# Circuit 15
# One repeatable layer structured as follows:
#   Block 1: RY on all qubits (4 parameters).
#   Block 2: CNOT gates:
#            CX(q3,q0), CX(q2,q3), CX(q1,q2), CX(q0,q1) (no parameters).
#   Block 3: RY on all qubits (4 parameters).
#   Block 4: CNOT gates:
#            CX(q3,q2), CX(q0,q3), CX(q1,q0), CX(q2,q1) (no parameters).
# Parameters per layer: 8 (4 + 4).

DESC_15  = "Circuit 15: RY on all qubits, CNOT block [q3->q0, q2->q3, q1->q2, q0->q1], RY on all qubits, CNOT block [q3->q2, q0->q3, q1->q0, q2->q1]. 8 parameters per layer."
PRE_LAYER_PARAMETERS_15 = 0
PARAMETERS_PER_LAYER_15 = 8

def circuit_15(qc, theta, n_layers):
    current_parameter = 0
    parameterized_gate_names = []

    for _ in range(n_layers):

        for i in range(4):
            qc.ry(theta[current_parameter], i)
            parameterized_gate_names.append('ry')
            current_parameter += 1

        qc.cx(3, 0)

        for i in range(3, 0, -1):
            qc.cx(i-1, i)

        for i in range(4):
            qc.ry(theta[current_parameter], i)
            parameterized_gate_names.append('ry')
            current_parameter += 1

        qc.cx(3, 2)

        qc.cx(0, 3)

        for i in range(2):
            qc.cx(i+1, i)

    return qc, parameterized_gate_names



# Circuit 16
# One repeatable layer structured as follows:
#   Block 1: RX then RZ on all qubits (8 parameters).
#   Block 2: two CRZ gates in parallel pairs:
#            CRZ(q1->q0) and CRZ(q3->q2) (2 parameters).
#   Block 3: one CRZ gate connecting the two pairs:
#            CRZ(q2->q1) (1 parameter).
# Parameters per layer: 11 (8 + 2 + 1).

DESC_16  = "Circuit 16: RX-RZ on all qubits, parallel controlled-RZ pairs (q1->q0, q3->q2), controlled-RZ (q2->q1). 11 parameters per layer."
PRE_LAYER_PARAMETERS_16 = 0
PARAMETERS_PER_LAYER_16 = 11

def circuit_16(qc, theta, n_layers):
    current_parameter = 0
    parameterized_gate_names = []

    for _ in range(n_layers):

        for i in range(4):
            qc.rx(theta[current_parameter], i)
            parameterized_gate_names.append('rx')
            current_parameter += 1

            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

        qc.crz(theta[current_parameter], 1, 0)
        parameterized_gate_names.append('crz')
        current_parameter += 1

        qc.crz(theta[current_parameter], 3, 2)
        parameterized_gate_names.append('crz')
        current_parameter += 1

        qc.crz(theta[current_parameter], 2, 1)
        parameterized_gate_names.append('crz')
        current_parameter += 1

    return qc, parameterized_gate_names



# Circuit 17
# One repeatable layer structured as follows:
#   Block 1: RX then RZ on all qubits (8 parameters).
#   Block 2: two CRX gates in parallel pairs:
#            CRX(q1->q0) and CRX(q3->q2) (2 parameters).
#   Block 3: one CRX gate connecting the two pairs:
#            CRX(q2->q1) (1 parameter).
# Identical structure to Circuit 16 with CRX replacing CRZ.
# Parameters per layer: 11 (8 + 2 + 1).

DESC_17  = "Circuit 17: RX-RZ on all qubits, parallel controlled-RX pairs (q1->q0, q3->q2), controlled-RX (q2->q1). 11 parameters per layer."
PRE_LAYER_PARAMETERS_17 = 0
PARAMETERS_PER_LAYER_17 = 11

def circuit_17(qc, theta, n_layers):
    current_parameter = 0
    parameterized_gate_names = []

    for _ in range(n_layers):

        for i in range(4):
            qc.rx(theta[current_parameter], i)
            parameterized_gate_names.append('rx')
            current_parameter += 1

            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

        qc.crx(theta[current_parameter], 1, 0)
        parameterized_gate_names.append('crx')
        current_parameter += 1

        qc.crx(theta[current_parameter], 3, 2)
        parameterized_gate_names.append('crx')
        current_parameter += 1

        qc.crx(theta[current_parameter], 2, 1)
        parameterized_gate_names.append('crx')
        current_parameter += 1

    return qc, parameterized_gate_names



# Circuit 18
# One repeatable layer structured as follows:
#   Block 1: RX on all qubits (4 parameters).
#   Block 2: RZ on all qubits (4 parameters).
#   Block 3: CRZ ring descending:
#            CRZ(q3->q0), CRZ(q2->q3), CRZ(q1->q2), CRZ(q0->q1) (4 parameters).
# Parameters per layer: 12 (4 + 4 + 4).

DESC_18  = "Circuit 18: RX on all qubits, RZ on all qubits, controlled-RZ descending ring. 12 parameters per layer."
PRE_LAYER_PARAMETERS_18 = 0
PARAMETERS_PER_LAYER_18 = 12

def circuit_18(qc, theta, n_layers):
    current_parameter = 0
    parameterized_gate_names = []

    for _ in range(n_layers):

        for i in range(4):
            qc.rx(theta[current_parameter], i)
            parameterized_gate_names.append('rx')
            current_parameter += 1

        for i in range(4):
            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

        qc.crz(theta[current_parameter], 3, 0)
        parameterized_gate_names.append('crz')
        current_parameter += 1

        for i in range(3, 0, -1):
            qc.crz(theta[current_parameter], i-1, i)
            parameterized_gate_names.append('crz')
            current_parameter += 1

    return qc, parameterized_gate_names



# Circuit 19
# One repeatable layer structured as follows:
#   Block 1: RX on all qubits (4 parameters).
#   Block 2: RZ on all qubits (4 parameters).
#   Block 3: CRX ring descending:
#            CRX(q3->q0), CRX(q2->q3), CRX(q1->q2), CRX(q0->q1) (4 parameters).
# Identical structure to Circuit 18 with CRX replacing CRZ.
# Parameters per layer: 12 (4 + 4 + 4).

DESC_19  = "Circuit 19: RX on all qubits, RZ on all qubits, controlled-RX descending ring. 12 parameters per layer."
PRE_LAYER_PARAMETERS_19 = 0
PARAMETERS_PER_LAYER_19 = 12

def circuit_19(qc, theta, n_layers):
    current_parameter = 0
    parameterized_gate_names = []

    for _ in range(n_layers):

        for i in range(4):
            qc.rx(theta[current_parameter], i)
            parameterized_gate_names.append('rx')
            current_parameter += 1

        for i in range(4):
            qc.rz(theta[current_parameter], i)
            parameterized_gate_names.append('rz')
            current_parameter += 1

        qc.crx(theta[current_parameter], 3, 0)
        parameterized_gate_names.append('crx')
        current_parameter += 1

        for i in range(3, 0, -1):
            qc.crx(theta[current_parameter], i-1, i)
            parameterized_gate_names.append('crx')
            current_parameter += 1

    return qc, parameterized_gate_names



# Registry mapping circuit index to its implementation.
# Index 0 is None so command-line circuit numbers match paper labels directly.
# Each entry: (function, fixed_params, layer_params, description).

CIRCUITS = [
    None,
    (circuit_1,  PRE_LAYER_PARAMETERS_1,  PARAMETERS_PER_LAYER_1,  DESC_1),
    (circuit_2,  PRE_LAYER_PARAMETERS_2,  PARAMETERS_PER_LAYER_2,  DESC_2),
    (circuit_3,  PRE_LAYER_PARAMETERS_3,  PARAMETERS_PER_LAYER_3,  DESC_3),
    (circuit_4,  PRE_LAYER_PARAMETERS_4,  PARAMETERS_PER_LAYER_4,  DESC_4),
    (circuit_5,  PRE_LAYER_PARAMETERS_5,  PARAMETERS_PER_LAYER_5,  DESC_5),
    (circuit_6,  PRE_LAYER_PARAMETERS_6,  PARAMETERS_PER_LAYER_6,  DESC_6),
    (circuit_7,  PRE_LAYER_PARAMETERS_7,  PARAMETERS_PER_LAYER_7,  DESC_7),
    (circuit_8,  PRE_LAYER_PARAMETERS_8,  PARAMETERS_PER_LAYER_8,  DESC_8),
    (circuit_9,  PRE_LAYER_PARAMETERS_9,  PARAMETERS_PER_LAYER_9,  DESC_9),
    (circuit_10, PRE_LAYER_PARAMETERS_10, PARAMETERS_PER_LAYER_10, DESC_10),
    (circuit_11, PRE_LAYER_PARAMETERS_11, PARAMETERS_PER_LAYER_11, DESC_11),
    (circuit_12, PRE_LAYER_PARAMETERS_12, PARAMETERS_PER_LAYER_12, DESC_12),
    (circuit_13, PRE_LAYER_PARAMETERS_13, PARAMETERS_PER_LAYER_13, DESC_13),
    (circuit_14, PRE_LAYER_PARAMETERS_14, PARAMETERS_PER_LAYER_14, DESC_14),
    (circuit_15, PRE_LAYER_PARAMETERS_15, PARAMETERS_PER_LAYER_15, DESC_15),
    (circuit_16, PRE_LAYER_PARAMETERS_16, PARAMETERS_PER_LAYER_16, DESC_16),
    (circuit_17, PRE_LAYER_PARAMETERS_17, PARAMETERS_PER_LAYER_17, DESC_17),
    (circuit_18, PRE_LAYER_PARAMETERS_18, PARAMETERS_PER_LAYER_18, DESC_18),
    (circuit_19, PRE_LAYER_PARAMETERS_19, PARAMETERS_PER_LAYER_19, DESC_19),
]
