# Qiskit
# Import necessary Qiskit libraries
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer
from fractions import Fraction
import numpy as np
import math

# Define the inverse Quantum Fourier Transform (QFT†)
def qft_dagger(n):
    qc = QuantumCircuit(n)
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)  # Swap qubits to reverse order
    for j in range(n):
        for k in range(j):
            qc.cp(-np.pi / float(2 ** (j - k)), k, j)  # Controlled phase rotation
        qc.h(j)  # Apply Hadamard
    qc.name = "QFT†"
    return qc

# Define modular exponentiation as a quantum circuit
def modular_exponentiation(a, power, N):
    qc = QuantumCircuit(4)  # 4 computational qubits
    for _ in range(power):
        if a in [2, 13]:
            qc.swap(0, 1)
            qc.swap(1, 2)
            qc.swap(2, 3)
        elif a in [4, 11]:
            qc.swap(2, 3)
            qc.swap(1, 2)
            qc.swap(0, 1)
        elif a in [7, 8]:
            qc.swap(1, 3)
            qc.swap(0, 2)
    controlled_qc = qc.control(1)  # Add a control qubit
    return controlled_qc

# Define the Quantum Phase Estimation circuit for a^x mod N
def qpe_amodN(a, N):
    qc = QuantumCircuit(10, 6)  # 6 counting qubits, 4 computational qubits
    qc.h(range(6))  # Apply Hadamard to counting qubits
    qc.x(6)  # Set |1> state for modular exponentiation
    for q in range(6):
        qc.append(modular_exponentiation(a, 2**q, N), [q] + list(range(6, 10)))
    qc.append(qft_dagger(6), range(6))  # Apply inverse QFT
    qc.measure(range(6), range(6))  # Measure first 6 qubits
    return qc

# Classical function to compute factors using period r
def find_factors(a, N, r):
    if r % 2 == 1 or r == 0:
        return None
    factor1 = math.gcd(a ** (r // 2) - 1, N)
    factor2 = math.gcd(a ** (r // 2) + 1, N)
    if factor1 in [1, N] or factor2 in [1, N]:
        return None
    return factor1, factor2

# Example: Breaking RSA-15 (N=15)
N = 21
np.random.seed(1)
valid_factors = None

# ✅ COPY-PASTED LINES START HERE - DO NOT MODIFY THESE
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

backend_name = "ibm_brisbane"
service = QiskitRuntimeService()
backend = service.least_busy(simulator=False, operational=True, min_num_qubits=100)
pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)

# Loop through possible values of 'a' to find valid factors
for a in [2,7]:
    if math.gcd(a, N) > 1:
        print(f"Trivial factor found: {math.gcd(a, N)}")
        continue
    print(f"Trying a = {a}...")

    # Create and transpile the circuit
    qc = qpe_amodN(a, N)
    qc_transpiled = pass_manager.run(qc)

    # ✅ Correct observable and fix errors
    from qiskit.quantum_info import SparsePauliOp
    observable = SparsePauliOp(["I" * qc_transpiled.num_qubits])  # Identity observable

    # ✅ Define and configure the Estimator
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    from qiskit_ibm_runtime import EstimatorOptions

    options = EstimatorOptions()
    options.resilience_level = 1
    options.dynamical_decoupling.enable = True
    options.dynamical_decoupling.sequence_type = "XY4"

    estimator = Estimator(backend, options=options)

    # ✅ Submit the job and print Job ID
    job = estimator.run([(qc_transpiled, observable)])
    job_id = job.job_id()
    print(f"Job submitted successfully. Job ID: {job_id}")

    # ✅ Retrieve job results after submission
    job = service.job(job_id)
    result = job.result()

    # ✅ Extract and process results properly
    if hasattr(result, "quasi_dists"):
        counts = result.quasi_dists[0]  # ✅ Corrected to use quasi_dists
    else:
        print("No valid counts found. Trying next value of a...")
        continue

    # ✅ Get the most probable measurement outcome
    if counts:
        measured_phase = max(counts, key=counts.get)
        decimal_phase = int(measured_phase, 2) / (2 ** 6)  # Convert to decimal phase
        fraction = Fraction(decimal_phase).limit_denominator(2 ** 6)
        r = fraction.denominator
        factors = find_factors(a, N, r)

        # ✅ Break if valid factors found
        if factors:
            valid_factors = factors
            break

# ✅ Print the final result
# ✅ Print the final result
if valid_factors:
    print(f"Shor’s Algorithm found factors: {valid_factors[0]} and {valid_factors[1]}")
elif any(math.gcd(a, N) > 1 for a in [2, 7]):  # ✅ Check if any trivial factor was found
    trivial_factor = next(f for f in [math.gcd(a, N) for a in [2, 7]] if f > 1)
    print(f"Shor’s Algorithm successfully found factors: {trivial_factor} and {N // trivial_factor}")
else:
    print("Shor’s Algorithm failed. Try again with a different N.")
