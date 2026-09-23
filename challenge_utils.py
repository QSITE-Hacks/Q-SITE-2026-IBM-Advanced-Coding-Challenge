# Adapted from IBM QDC 2025 challenge_utils.py (qiskit-community/qdc-challenges-2025, Apache-2.0).
# Modifications for the UofT Qiskit Fall Fest edition are marked with "# FF:".


from qiskit.visualization import plot_gate_map
from qiskit import QuantumCircuit
import numpy as np
from qiskit.circuit.gate import Gate

def barbell(a1: float, a2: float, a3: float, a4: float, a5: float, a6: float) -> Gate:
    """
    Constructs a gate implementing barbell circuit as visualized above (see Figure 8)

    Args:
        a1, a2, a3, a4, a5, a6: circuit parameters
    
    Returns:
        Gate implementing the unitary defined by the barbell circuit
    """

    qc = QuantumCircuit(4, name='barbell')
    
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    qc.rz(a1, 1)
    qc.rz(a4, 2)
    qc.rz(a6, 3)

    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(0, 1)

    qc.rz(a2, 2)
    qc.rz(a5, 3)

    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.rz(a3, 3)
    qc.cx(2, 3)
    
    gate = qc.to_gate(label=fr'barbell ({a1}, {a2}, {a3}, {a4}, {a5}, {a6})')
    return gate

def trotter_step_electric_2q(circ: QuantumCircuit, L: int, t: float, g: float) -> QuantumCircuit:
    """
    Adds Trotter step component corresponding to the electric part of the Hamiltonian to input circuit. Copied directly from Appendix D.

    Args:
        circ: input circuit
        L: number of spatial sites
        t: evolution time for the Trotter step
        g: coupling constant
    
    Returns:
        Circuit with the electric part of the Trotter step added
    """
    
    if np.floor(L/4) == np.floor((L-2)/4):
        for n in range(0,int((L-2)/2),2):
            if n==0:
                a1=(g**2*t)*(1+n*4)/4
                a2=(g**2*t)*(3+n*4)/4
                a3=0
                a4=a1
                a5=a2
                a6=a1
                circ.append(barbell(a1,a2,a3,a4,a5,a6), [L+2*n,L+1+2*n,L+2+2*n,L+3+2*n])
                circ.append(barbell(a3,a2,a1,a5,a4,a6), [L-4-2*n,L-3-2*n,L-2-2*n,L-1-2*n])
            else:
                a1=0
                a2=(g**2*t)*(3+n*4)/4
                a3=0
                a4=(g**2*t)*(1+n*4)/4
                a5=a2
                a6=a4
                circ.append(barbell(a1,a2,a3,a4,a5,a6), [L+2*n,L+1+2*n,L+2+2*n,L+3+2*n])
                circ.append(barbell(a3,a2,a1,a5,a4,a6), [L-4-2*n,L-3-2*n,L-2-2*n,L-1-2*n])
        for n in range(1,int((L-2)/2),2):
            a1=(g**2*t)*(1+n*4)/4
            a2=(g**2*t)*(3+n*4)/4
            a3=(g**2*t)*(5+n*4)/4
            a4=a1
            a5=a2
            a6=a1
            circ.append(barbell(a1,a2,a3,a4,a5,a6), [L+2*n,L+1+2*n,L+2+2*n,L+3+2*n])
            circ.append(barbell(a3,a2,a1,a5,a4,a6), [L-4-2*n,L-3-2*n,L-2-2*n,L-1-2*n])
    else:
        for n in range(0,int((L-2)/2),2):
            a1=(g**2*t)*(1+n*4)/4
            a2=(g**2*t)*(3+n*4)/4
            a3=(g**2*t)*(5+n*4)/4
            a4=a1
            a5=a2
            a6=a1
            circ.append(barbell(a1,a2,a3,a4,a5,a6), [L+2*n,L+1+2*n,L+2+2*n,L+3+2*n])
            circ.append(barbell(a3,a2,a1,a5,a4,a6), [L-4-2*n,L-3-2*n,L-2-2*n,L-1-2*n])
        for n in range(1,int((L-2)/2),2):
            a1=0
            a2=(g**2*t)*(3+n*4)/4
            a3=0
            a4=(g**2*t)*(1+n*4)/4
            a5=a2
            a6=a4
            circ.append(barbell(a1,a2,a3,a4,a5,a6), [L+2*n,L+1+2*n,L+2+2*n,L+3+2*n])
            circ.append(barbell(a3,a2,a1,a5,a4,a6), [L-4-2*n,L-3-2*n,L-2-2*n,L-1-2*n])
    
    return circ

def RXXplus(theta: float) -> Gate:
    """
    Constructs a gate implementing XX+YY rotation as visualized above

    Args:
        theta: rotation angle
    
    Returns:
        Gate implementing the unitary
    """
    
    qc = QuantumCircuit(2, name='RXXplus')
    
    qc.h(0)
    qc.s(0)
    qc.h(1)
    qc.s(1)
    
    qc.cx(0, 1)
    qc.h(0)
    qc.rz(theta, 0)
    qc.rz(theta, 1)
    qc.h(0)
    qc.cx(0, 1)
    
    qc.sdg(0)
    qc.h(0)
    qc.sdg(1)
    qc.h(1)
    
    gate = qc.to_gate(label=fr'$R^{{XX}}_{{+}}({theta})$')
    return gate

def postselection_and_mitigation(chi, chi_cal, chi_exact, L, suppression_threshold=0.01):

    chi_mitigated = np.zeros((2 * L))

    suppression_factors = (1 - chi_cal)/(1 - chi_exact)
    # sum over the half of the lattice:
    for q in range(L):
        
        chi_selected = []
        suppression_factors_selected = []

        # Site q:
        suppression_factor = suppression_factors[q]
        if suppression_factor > suppression_threshold:
            chi_selected.append(chi[q])
            suppression_factors_selected.append(suppression_factor)

        # The CP/mirror image of q:
        suppression_factor = suppression_factors[2*L-1-q]
        if suppression_factor > suppression_threshold:
            chi_selected.append(chi[2*L-1-q])
            suppression_factors_selected.append(suppression_factor)

        if len(chi_selected) > 0:
            suppression_factors_selected = np.array(suppression_factors_selected)
            chi_selected = np.array(chi_selected)
            chi_mitigated[q] = 1 - (1 - chi_selected.mean())/suppression_factors_selected.mean()
        else:
            chi_mitigated[q] = np.nan
    
    chi_mitigated[2*L-1:L-1:-1] = chi_mitigated[:L]

    return chi_mitigated

# Backend coupling map related utils

def plot_qubit_chain(qubit_chain, backend, qubit_coordinates):
    qubits = range(backend.target.num_qubits)
    qcolors = []
    q_lattice = 0
    for q in range(len(qubits)):
        if q in qubit_chain:
            if np.where(np.array(qubit_chain) == q)[0] % 2 == 0:
                qcolors.append('blue')
            else:
                qcolors.append('orange')
            q_lattice += 1
        else:
            qcolors.append('grey')
    display(plot_gate_map(backend, qubit_coordinates=qubit_coordinates, qubit_color=qcolors)) #, label_qubits=True, , line_color=["black" for _ in backend.coupling_map.get_edges()])

def find_lowest_connected_qubit(q, edges):

    connected_qubits = []
    for edge in edges:
        if edge[0] == q:
            connected_qubits.append(edge[1])
        if edge[1] == q:
            connected_qubits.append(edge[0])

    return min(connected_qubits)

def get_qubit_coordinates(backend, width=None):
    """
    get qubit coordinates for coupling map
    currently only works for Herons!
    """
    qubit_coordinates = []
    edges = list(backend.coupling_map.get_edges())
    qubits = range(backend.target.num_qubits)
    
    connector_width = 4
    if not width:
        # FF: infer the heavy-hex row width from the qubit count so that real and fake
        # Heron backends (ibm_* and fake_*) both work: 156 qubits -> 16 per row, 133 -> 15.
        nq = backend.target.num_qubits
        if nq == 156:
            width = 16
        elif nq == 133:
            width = 15
        else:
            raise ValueError(f"{backend.name} ({nq} qubits) is not a Heron layout; pass width= explicitly")


    row_type = 0 # 0 is full row, 1 is row of connector qubits
    ix = 0
    iy = 0
    for q in qubits:
        if row_type == 1:
            q_above = find_lowest_connected_qubit(q, edges)
            x = qubit_coordinates[q_above][1]
            qubit_coordinates.append([iy, x])
        else:
            qubit_coordinates.append([iy, ix])
        if row_type == 0:
            if ix < width - 1:
                ix += 1
            else:
                row_type = 1
                ix = 0
                iy -= 1
        else:
            if ix < connector_width - 1:
                ix += 1
            else:
                row_type = 0
                ix = 0
                iy -= 1
    return qubit_coordinates
