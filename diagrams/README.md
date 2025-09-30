## Diagram descriptions

### Quantum embedding (QuantumBeliefState)
Input state `s ∈ R^{state_dim}` is first linearly projected to match the qubit count `n_qubits`. The projected vector is angle-encoded across all qubits. A variational circuit with `StronglyEntanglingLayers` of depth `L` (trainable weights shaped `(L, n_qubits, 3)`) processes the encoded state. The circuit outputs classical features by measuring expectation values `⟨Z_i⟩` on each qubit, giving `z ∈ R^{n_qubits}`. A classical MLP then post-processes `z` to produce the quantum belief vector `b(s) ∈ R^{32}`. Implementation corresponds to `QuantumBeliefState` using a PennyLane QNode wrapped via `qml.qnn.TorchLayer` on `default.qubit`.

### Hybrid encoder (HybridBeliefState)
The hybrid pathway branches into classical and quantum encoders from the same input `s`. The classical branch is a compact MLP producing `b_class(s) ∈ R^{classical_dim}`. The quantum branch mirrors the quantum embedding pipeline: projection → angle encoding → `StronglyEntanglingLayers` → `⟨Z_i⟩` → post-MLP yielding `b_quant(s) ∈ R^{32}`. These two beliefs are concatenated and passed through a fusion MLP to obtain the hybrid belief `b_hybrid(s) ∈ R^{classical_dim+32}`. Optionally, an update block concatenates `b_hybrid` with the raw state and refines it via another MLP. This reflects `HybridBeliefState` in code, combining classical and quantum components before fusion and (optionally) belief update.


