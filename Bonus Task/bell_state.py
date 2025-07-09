from qiskit import QuantumCircuit
from qiskit_aer.primitives import Sampler
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Create a Bell State circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Run using Aer Sampler (simulator)
sampler = Sampler()
result = sampler.run(qc, shots=1000).result()

# Extract and print results
counts = result.quasi_dists[0].binary_probabilities()
print("Bell state results:", counts)

# Plot
plot_histogram(counts)
plt.title("Bell State Simulation with Sampler")
plt.show()
