# Tutorial to Coursework Concept Map

This document maps concepts from the tutorial notebooks to specific coursework tasks.

## Week 1: Neurons (w1-neurons-solution.ipynb)

### Key Concepts
- **Leaky Integrate-and-Fire (LIF) neuron model**
- **Membrane potential dynamics**: `dv/dt = (I - v) / tau`
- **Discrete simulation**: Euler integration
- **Spike detection and reset**

### Used in Coursework
- **Task 3A**: Implementing `SNNLayer` class
  - LIF neuron dynamics in `forward()` method
  - Membrane potential update: `v_new = alpha * v + (1 - alpha) * I`
  - Spike detection and reset logic
  - Time constant `tau` parameter

- **Task 3B**: Verification
  - Testing LIF neuron behavior
  - Comparing simulation to analytic solution
  - Understanding how time constants affect neuron behavior

### Code Patterns to Reuse
```python
# From Week 1: LIF simulation loop
alpha = np.exp(-dt/tau)
V = alpha*V + (1-alpha)*I[t_idx]
if V > threshold:
    V = reset
    spikes.append(t_idx*dt)
```

**Adapted for Coursework**:
```python
# In SNNLayer.forward()
alpha = torch.exp(-dt / self.tau)
v = alpha * v + (1 - alpha) * I
spikes = surrogate_heaviside(v - threshold)
v = v * (1 - spikes) + reset * spikes
```

---

## Week 2: Synapses and Networks (w2-synapses-networks-solution.ipynb)

### Key Concepts
- **Synaptic dynamics**: Biexponential synapses
- **Network connectivity**: All-to-all connections
- **Multiple neurons**: Batch processing
- **Temporal dynamics**: Delays, integration over time

### Used in Coursework
- **Task 3A**: Network architecture
  - Weight matrices for connectivity
  - Multiple neurons processing simultaneously
  - Batch processing (multiple examples at once)
  - Input current computation: `I = torch.matmul(x, w)`

- **Task 4**: Multi-layer networks
  - Chaining layers together
  - Hidden layers and output layers
  - Different layer types (spiking vs non-spiking)

### Code Patterns to Reuse
```python
# From Week 2: Network simulation
# Processing multiple neurons, time steps
for t_idx in range(num_time_steps):
    # Update all neurons at once
    v = update_neurons(v, I, tau, dt)
```

**Adapted for Coursework**:
```python
# In SNNLayer.forward()
for t in range(num_time_points):
    I = torch.matmul(x[:, :, t], self.w)  # Batch processing
    v = alpha * v + (1 - alpha) * I  # Vectorized update
```

---

## Week 3: Brain Structure (w3-exercise-solution.ipynb)

### Key Concepts
- **Data analysis**: Loading and processing real neural data
- **Visualization**: Plotting neural data
- **Statistics**: Computing rates, distributions
- **Data exploration**: Understanding data structure

### Used in Coursework
- **Task 1A**: Data preprocessing
  - Loading `.mat` files (similar to loading neuron morphologies)
  - Computing statistics (number of neurons, spikes, duration)
  - Data normalization (whitening)

- **Task 1B**: Data visualization
  - Raster plots (similar to plotting neuron morphologies)
  - Time series plots
  - Histograms and bar charts
  - Multi-panel figures

- **Task 4**: Network analysis
  - Computing firing rates
  - Plotting spike trains
  - Analyzing network activity

### Code Patterns to Reuse
```python
# From Week 3: Data loading and analysis
# Loading data, computing statistics, plotting
neuron_paths = load_neurons(folder_path)
for neuron in neurons:
    # Process and analyze
    plot_neuron(neuron)
```

**Adapted for Coursework**:
```python
# In Task 1: Loading and analyzing spike data
spike_times = load_spike_data('s1_data_raw.mat')
firing_rates = [len(st) / duration for st in spike_times]
plot_raster(spike_times)
```

---

## Past Coursework Sample (synfire-answers.ipynb)

### Key Concepts
- **Complete implementation**: Full working code
- **Code structure**: Modular functions, classes
- **Documentation**: Clear comments and explanations
- **Testing**: Verification of implementation
- **Visualization**: Comprehensive plotting

### Used in Coursework
- **Task 2B**: Data batching
  - Generator functions (similar pattern to synfire simulation)
  - Efficient data loading
  - Batch processing

- **Task 3A**: Network implementation
  - Class-based design (similar to synfire simulator)
  - Modular code structure
  - Parameter initialization
  - Forward pass implementation

- **Task 5**: Training loop
  - Iterating over batches
  - Loss computation
  - Optimization
  - Progress tracking

### Code Patterns to Reuse
```python
# From Past Coursework: Generator function
def simulate_layer(...):
    for t_idx in range(num_time_steps):
        # Simulation code
        yield results
```

**Adapted for Coursework**:
```python
# In Task 2B: Data generator
def batched_data(...):
    for batch_idx in range(num_batches):
        x = create_batch(...)
        y = create_targets(...)
        yield x, y
```

---

## Concept Dependency Graph

```
Week 1 (LIF Neurons)
    ↓
Task 3A (SNNLayer) ──→ Task 3B (Verification)
    ↓
Task 4 (Evaluation) ──→ Task 5 (Training)
    ↓
Task 6 (Decoding) ──→ Task 7 (Comparison)

Week 2 (Networks)
    ↓
Task 3A (Multi-layer) ──→ Task 4 (Network Architecture)

Week 3 (Data Analysis)
    ↓
Task 1A (Preprocessing) ──→ Task 1B (Visualization)
    ↓
Task 2A (Analysis) ──→ Task 2B (Batching)

Past Coursework (Structure)
    ↓
All Tasks (Code Organization, Documentation)
```

---

## Key Differences: Tutorials vs Coursework

### Tutorials (Weeks 1-3)
- **NumPy-based**: Uses NumPy arrays
- **Sequential processing**: One example at a time
- **Fixed parameters**: Time constants, weights are constants
- **Simple visualization**: Basic plots
- **Educational focus**: Learning concepts

### Coursework
- **PyTorch-based**: Uses PyTorch tensors
- **Batch processing**: Multiple examples simultaneously
- **Trainable parameters**: Weights and time constants are learnable
- **Complex visualization**: Multi-panel figures, loss curves
- **Research focus**: Real data, training, evaluation

### Migration Path
1. **NumPy → PyTorch**: Replace `np.array` with `torch.Tensor`
2. **Loops → Vectorization**: Use tensor operations
3. **Fixed → Trainable**: Use `nn.Parameter` for learnable values
4. **Forward → Backward**: Add gradient computation with `backward()`
5. **Manual → Automatic**: Use PyTorch's autograd

---

## Study Guide: What to Review

### Before Starting Task 1
- Review Week 3: Data loading and visualization
- Understand NumPy array operations
- Know how to plot time series data

### Before Starting Task 2
- Review Week 3: Data analysis techniques
- Understand interpolation (`np.interp`)
- Know how to create generator functions

### Before Starting Task 3
- **Critical**: Review Week 1: LIF neuron implementation
- Review Week 2: Network connectivity
- Understand PyTorch basics (`nn.Module`, `nn.Parameter`)
- Understand tensor operations

### Before Starting Task 4
- Review Task 3: SNNLayer implementation
- Understand evaluation metrics (MSE)
- Know how to visualize network internals

### Before Starting Task 5
- Review Past Coursework: Training loops
- Understand PyTorch optimizers (`torch.optim.Adam`)
- Understand loss functions and backpropagation
- Know how to track training progress

### Before Starting Task 6
- Review Task 5: Trained network usage
- Understand data batching and windowing
- Know how to create multi-panel plots

### Before Starting Task 7
- Review Task 5: Training procedure
- Understand spiking vs non-spiking differences
- Know how to compare model performance

---

## Common Patterns Across Tutorials and Coursework

### 1. Simulation Loop
```python
# Pattern: Initialize → Loop over time → Update → Record
v = initialize()
for t in range(num_steps):
    v = update(v, input[t])
    record(v, t)
```

### 2. Data Processing
```python
# Pattern: Load → Preprocess → Analyze → Visualize
data = load_data()
data = preprocess(data)
stats = analyze(data)
plot(data, stats)
```

### 3. Network Forward Pass
```python
# Pattern: Input → Layers → Output
x = input
for layer in layers:
    x = layer(x)
output = x
```

### 4. Training Loop
```python
# Pattern: Forward → Loss → Backward → Update
for epoch in epochs:
    for batch in batches:
        output = model(batch)
        loss = compute_loss(output, target)
        loss.backward()
        optimizer.step()
```

---

## Resources by Task

### Task 1: Data Loading
- Week 3: Loading and processing data
- Scipy: `io.loadmat()` for MATLAB files
- NumPy: Array operations, statistics

### Task 2: Data Batching
- Week 3: Data analysis
- Python: Generator functions (`yield`)
- Scipy: `interpolate` for velocity interpolation

### Task 3: SNN Implementation
- **Week 1**: LIF neuron dynamics (CRITICAL)
- Week 2: Network connectivity
- PyTorch: `nn.Module`, `nn.Parameter`, autograd
- Surrogate gradients: Provided in coursework

### Task 4: Evaluation
- Week 3: Visualization techniques
- PyTorch: `torch.no_grad()` for evaluation
- Matplotlib: Multi-panel figures

### Task 5: Training
- Past Coursework: Training loop structure
- PyTorch: Optimizers, loss functions, backpropagation
- Tqdm: Progress bars

### Task 6: Decoding
- Task 5: Using trained models
- Task 2: Data batching and windowing
- Matplotlib: Complex multi-panel plots

### Task 7: Comparison
- Task 5: Training procedure
- Task 6: Evaluation and visualization
- Statistical analysis: Comparing distributions

---

## Quick Reference: Tutorial Code → Coursework Adaptation

### Week 1: LIF Neuron
```python
# Tutorial (NumPy)
def LIF(I, tau=10, threshold=1.0, reset=0.0, dt=0.1):
    alpha = np.exp(-dt/tau)
    V = 0.0
    for t_idx in range(len(I)):
        V = alpha*V + (1-alpha)*I[t_idx]
        if V > threshold:
            V = reset
```

```python
# Coursework (PyTorch)
class SNNLayer(nn.Module):
    def forward(self, x):
        alpha = torch.exp(-dt / self.tau)
        v = torch.zeros(batch_size, n_out)
        for t in range(num_time_points):
            I = torch.matmul(x[:, :, t], self.w)
            v = alpha * v + (1 - alpha) * I
            spikes = surrogate_heaviside(v - threshold)
            v = v * (1 - spikes) + reset * spikes
```

### Week 2: Network Simulation
```python
# Tutorial: Multiple neurons
v = np.zeros(num_neurons)
for t_idx in range(num_time_steps):
    I = compute_input(spikes_in[t_idx])
    v = update_neurons(v, I, tau, dt)
```

```python
# Coursework: Batch processing
v = torch.zeros(batch_size, n_out)
for t in range(num_time_points):
    I = torch.matmul(x[:, :, t], self.w)  # Vectorized
    v = alpha * v + (1 - alpha) * I
```

### Week 3: Data Analysis
```python
# Tutorial: Loading and analyzing
neurons = load_neurons(path)
rates = [compute_rate(n) for n in neurons]
plot_histogram(rates)
```

```python
# Coursework: Spike data analysis
spike_times = load_spike_data('s1_data_raw.mat')
firing_rates = [len(st) / duration for st in spike_times]
plt.barh(range(len(spike_times)), firing_rates)
```

---

This map should help you understand which tutorial concepts are needed for each coursework task. Refer back to the relevant tutorials when working on each task!

