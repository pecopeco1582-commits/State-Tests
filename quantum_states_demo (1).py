"""
Qiskit program demonstrating W state and GHZ state probabilities
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt


def create_w_state(num_qubits=3):
    """
    Create a W state: equal superposition of single excitations
    For 3 qubits: (|100> + |010> + |001>) / sqrt(3)
    Each basis state appears with probability 1/3

    Construction:
      Start with |100> (X on qubit 0).
      At each step i, qubit i holds all remaining amplitude.
      A controlled-RY (decomposed as RY(t/2)·CX·RY(-t/2)) splits
      1/(n-i) of amplitude onto qubit i+1, then a CNOT passes the
      excitation down the chain. This strictly preserves the
      single-excitation subspace — no |000>, |111>, etc. can appear.
    """
    import numpy as np

    qc = QuantumCircuit(num_qubits, num_qubits, name='W State')

    # Initialise to |100...0>
    qc.x(0)

    for i in range(num_qubits - 1):
        # At step i, qubit i holds amplitude for (n-i) remaining slots.
        # We want to leave (n-i-1)/(n-i) on qubit i and send 1/(n-i) forward.
        # The RY angle on the target that achieves this:
        theta = 2 * np.arcsin(np.sqrt(1.0 / (num_qubits - i)))

        # Controlled-RY(theta) on qubit i+1, controlled by qubit i
        # Decomposition: RY(theta/2) · CNOT · RY(-theta/2)
        qc.ry(theta / 2, i + 1)
        qc.cx(i, i + 1)
        qc.ry(-theta / 2, i + 1)

        # CNOT to pass the |1> excitation down the chain
        qc.cx(i + 1, i)

    qc.measure(range(num_qubits), range(num_qubits))
    return qc


# Removed: Alternative W state method (using standard construction above)


def create_ghz_state(num_qubits=3):
    """
    Create a GHZ state: (|000> + |111>) / sqrt(2)
    Measuring will give either all 0s or all 1s with equal probability
    """
    qc = QuantumCircuit(num_qubits, num_qubits, name='GHZ State')
    
    # Apply Hadamard to first qubit to create superposition
    qc.h(0)
    
    # Apply CNOT gates to entangle all qubits
    for i in range(num_qubits - 1):
        qc.cx(0, i + 1)
    
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def run_simulation(circuit, shots=1000):
    """
    Run quantum circuit simulation and return results
    """
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(circuit)
    return counts


def plot_results(w_counts, ghz_counts):
    """
    Plot and display results for both W state and GHZ state
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot W state results
    states_w = list(w_counts.keys())
    probs_w = [w_counts[state] / sum(w_counts.values()) for state in states_w]
    
    axes[0].bar(states_w, probs_w, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Basis State')
    axes[0].set_ylabel('Probability')
    axes[0].set_title('W State Probabilities\n(Expected: 1/3 ≈ 0.333 for |001>, |010>, |100>)')
    axes[0].set_ylim([0, 0.5])
    axes[0].axhline(y=1/3, color='r', linestyle='--', label='1/3')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot GHZ state results
    states_ghz = list(ghz_counts.keys())
    probs_ghz = [ghz_counts[state] / sum(ghz_counts.values()) for state in states_ghz]
    
    axes[1].bar(states_ghz, probs_ghz, color='forestgreen', alpha=0.7)
    axes[1].set_xlabel('Basis State')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('GHZ State Probabilities\n(Expected: 1/2 ≈ 0.5 for |000> and |111>)')
    axes[1].set_ylim([0, 0.6])
    axes[1].axhline(y=1/2, color='r', linestyle='--', label='1/2')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('quantum_states_results.png', dpi=150)
    plt.show()


def print_analysis(w_counts, ghz_counts, shots):
    """
    Print detailed analysis of results
    """
    print("=" * 70)
    print("QUANTUM STATE PROBABILITY ANALYSIS")
    print("=" * 70)
    
    # W State Analysis
    print("\n[W STATE ANALYSIS]")
    print("-" * 70)
    print("The W state is a superposition of three single-excitation states:")
    print("W = (|100⟩ + |010⟩ + |001⟩) / √3")
    print("\nExpected probabilities: Each state → 1/3 ≈ 0.3333")
    print(f"\nSimulation results ({shots} shots):")
    
    total_w = sum(w_counts.values())
    w_states = sorted(w_counts.keys())
    for state in w_states:
        prob = w_counts[state] / total_w
        expected = 1/3
        error = abs(prob - expected) / expected * 100
        print(f"  {state}: {w_counts[state]:4d} counts → {prob:.4f} " + 
              f"(Expected: {expected:.4f}, Error: {error:.2f}%)")
    
    # GHZ State Analysis
    print("\n[GHZ STATE ANALYSIS]")
    print("-" * 70)
    print("The GHZ state is a maximally entangled state:")
    print("GHZ = (|000⟩ + |111⟩) / √2")
    print("\nExpected probabilities:")
    print("  |000⟩ → 1/2 = 0.5")
    print("  |111⟩ → 1/2 = 0.5")
    print(f"\nSimulation results ({shots} shots):")
    
    total_ghz = sum(ghz_counts.values())
    ghz_states = sorted(ghz_counts.keys())
    for state in ghz_states:
        if state in ghz_counts:
            prob = ghz_counts[state] / total_ghz
            expected = 1/2
            error = abs(prob - expected) / expected * 100
            print(f"  {state}: {ghz_counts[state]:4d} counts → {prob:.4f} " + 
                  f"(Expected: {expected:.4f}, Error: {error:.2f}%)")
        else:
            print(f"  {state}: 0 counts (not observed)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Number of qubits
    num_qubits = 3
    shots = 1000
    
    print(f"\nCreating quantum circuits for {num_qubits} qubits...")
    
    # Create circuits
    w_circuit = create_w_state(num_qubits)
    ghz_circuit = create_ghz_state(num_qubits)
    
    # Display circuits
    print("\nW State Circuit:")
    print(w_circuit)
    print("\n" + "=" * 70)
    print("\nGHZ State Circuit:")
    print(ghz_circuit)
    print("\n" + "=" * 70)
    
    # Run simulations
    print(f"\nRunning simulations ({shots} shots)...")
    w_counts = run_simulation(w_circuit, shots=shots)
    ghz_counts = run_simulation(ghz_circuit, shots=shots)
    
    # Print analysis
    print_analysis(w_counts, ghz_counts, shots)
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(w_counts, ghz_counts)
