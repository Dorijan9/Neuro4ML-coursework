# Neuro4ML Coursework Breakdown: Neural Decoding with Spiking Neural Networks

## Overview
**Goal**: Build a spiking neural network (SNN) decoder that predicts monkey arm velocity from motor cortex spike recordings using surrogate gradient descent.

**Data**: Spike trains from monkey motor cortex while performing a pointer movement task, plus recorded velocity data (x, y components).

**Key Concept**: This is a **brain-machine interface** - using raw neural spikes to predict behavior (velocity).

---

## Coursework Structure & Dependencies

### Prerequisites from Tutorials
- **Week 1 (w1-neurons-solution.ipynb)**: Leaky Integrate-and-Fire (LIF) neuron models
- **Week 2 (w2-synapses-networks-solution.ipynb)**: Synapses, networks, temporal dynamics
- **Week 3 (w3-exercise-solution.ipynb)**: Data analysis, connectivity
- **Past Coursework Sample**: Reference for structure and coding style

---

## Task Breakdown

### **TASK 1: Load and Plot the Data** ⚠️

#### Task 1A: Preprocess and Compute Basic Statistics
**Status**: Not started  
**What to do**:
1. **Whiten velocities**: Transform so mean=0, std=1
   - Formula: `vel_whitened = (vel - mean(vel)) / std(vel)`
   - Do this separately for x and y components

2. **Count neurons and spikes**:
   - `num_neurons = len(spike_times)`
   - `total_spikes = sum(len(st) for st in spike_times)`

3. **Compute experiment duration**:
   - `duration = max(max(st) for st in spike_times)` (from spikes)
   - Or `duration = vel_times[-1] - vel_times[0]` (from velocities)

4. **Estimate spike sampling rate**:
   - Spikes are event-based (not regularly sampled)
   - Can estimate from minimum time between spikes or look up in paper
   - Paper suggests ~30 kHz sampling (but this is recording resolution, not spike rate)

5. **Estimate velocity sampling rate**:
   - `velocity_sampling_rate = len(vel_times) / (vel_times[-1] - vel_times[0])`
   - Typically ~60 Hz for this type of data

**Dependencies**: None (data loading code provided)

---

#### Task 1B: Plot the Data
**Status**: Not started  
**What to do**:
1. **Raster plot (whole dataset)**:
   - For each neuron `i`, plot spikes at `(spike_time, neuron_index)`
   - Use `plt.scatter()` or `plt.eventplot()`

2. **Raster plot (1000-1010 seconds)**:
   - Filter spikes: `spikes_in_window = [st[(st >= 1000) & (st <= 1010)] for st in spike_times]`

3. **Velocity plots (whole + windowed)**:
   - Plot `vel[:, 0]` (x) and `vel[:, 1]` (y) vs `vel_times`
   - Use different colors/linestyles for x and y

4. **Firing rate bar chart**:
   - `firing_rates = [len(st) / duration for st in spike_times]`
   - Use `plt.barh()` to show rates for each neuron

5. **Velocity trajectory in (x,y) space**:
   - Plot `vel[:, 0]` vs `vel[:, 1]`
   - Highlight region from t=1000 to t=1010 with different color

**Dependencies**: Task 1A (need whitened data and statistics)

**Template provided**: Yes (commented code in notebook)

---

### **TASK 2: Divide Data into Test/Train and Batches** ⚠️

#### Task 2A: Why Not Continuous Time Ranges?
**Status**: Not started  
**What to do**:
1. **Compute speed**: `speed = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)`
2. **Plot speed over time** (with windowing/averaging for clarity):
   - Use moving average or `savgol_filter()` to smooth
3. **Show statistics**:
   - Print mean, std, min, max speeds
   - Show that speed varies significantly over time
   - **Why this matters**: If you split 70/15/15 by time, training might only see slow movements and testing only fast movements → poor generalization

**Dependencies**: Task 1A (need velocity data)

---

#### Task 2B: Equally Distributed Segments
**Status**: Not started  
**What to do**:

1. **Divide data into segments**:
   - Split time range into equal segments (e.g., 1 second each)
   - Assign segments to train/val/test sets (e.g., 70% train, 15% val, 15% test)
   - **Important**: Distribute segments throughout the entire time range (not consecutive blocks)
   - Example: If you have 100 segments, use segments [0, 3, 6, 9, ...] for train, [1, 4, 7, 10, ...] for val, [2, 5, 8, 11, ...] for test

2. **Write `batched_data()` generator function**:
   - **Inputs**: `range_to_use` (time ranges), `dt=1e-3`, `length=1`, `batch_size=64`
   - **Outputs**: `(x, y)` tuples
   - **x shape**: `(batch_size, num_neurons, num_time_points)`
     - `num_time_points = int(length / dt)` (e.g., 1000 for 1 second at 1ms)
     - Convert spike times to binary array (1 = spike, 0 = no spike)
   - **y shape**: `(batch_size, 2, num_time_points)`
     - Interpolate velocities to match simulation time points
     - Use `np.interp()` or `scipy.interpolate` to get velocities at simulation times

3. **Plot sample batch**:
   - Get one batch: `x, y = next(batched_data(...))`
   - Plot spikes and velocities for 4 samples from the batch

**Key Implementation Details**:
- **Spike conversion**: For each time bin `t`, check if any spike falls in `[t, t+dt)`
- **Velocity interpolation**: `np.interp(sim_times, vel_times, vel[:, 0])` for x-component
- **Randomization**: Shuffle segments within each epoch for training

**Dependencies**: Task 1A, Task 2A

**Template provided**: Yes (commented code)

---

### **TASK 3: Spiking Neural Network Model** ⚠️

#### Task 3A: Single Layer Simulation Code
**Status**: Not started  
**What to do**:

1. **Create `SNNLayer` class** (derived from `nn.Module`):
   - **Inputs**: `n_in`, `n_out`, `spiking=True`
   - **Parameters**:
     - `self.w`: Weight matrix `(n_in, n_out)` - trainable
     - `self.tau`: Time constants `(n_out,)` - trainable, per-neuron
     - Threshold, reset values (can be fixed or trainable)
   - **Initialization**:
     - Weights: Small random values (e.g., `torch.randn() * 0.1`)
     - Time constants: Uniform distribution (20-100 ms for spiking, 200-1000 ms for non-spiking)
     - Make `tau` a parameter: `self.tau = nn.Parameter(torch.rand(n_out) * (tau_max - tau_min) + tau_min)`

2. **Implement `forward()` method**:
   - **Input**: `x` shape `(batch_size, num_input_neurons, num_time_points)`
   - **Output**: `y` shape `(batch_size, num_output_neurons, num_time_points)`
   - **Algorithm**:
     ```python
     # Initialize membrane potentials
     v = torch.zeros(batch_size, n_out)
     
     # Loop over time steps
     for t in range(num_time_points):
         # Compute input current: weighted sum of input spikes
         I = torch.matmul(x[:, :, t], self.w)  # (batch_size, n_out)
         
         # Update membrane potential (LIF dynamics)
         # dv/dt = (I - v) / tau
         # Discrete: v_new = v * exp(-dt/tau) + I * (1 - exp(-dt/tau))
         alpha = torch.exp(-dt / self.tau)  # (n_out,)
         v = alpha * v + (1 - alpha) * I
         
         # Check for spikes
         if self.spiking:
             spikes = surrogate_heaviside(v - threshold)
             v = v * (1 - spikes) + reset * spikes  # Reset after spike
             y[:, :, t] = spikes
         else:
             y[:, :, t] = v
     ```

3. **Create multi-layer network class** (optional but recommended):
   - `SNNNetwork(nn.Module)` that chains multiple `SNNLayer` instances
   - Useful for Task 4 and beyond

**Key Points**:
- Use `surrogate_heaviside` (provided) for spiking output
- Ensure `tau` stays positive (clamp or use `torch.abs()` or `torch.exp()`)
- Time constants are per-neuron (heterogeneous)

**Dependencies**: Surrogate gradient function (provided), PyTorch basics

**Template provided**: Yes (commented class structure)

---

#### Task 3B: Verify Your Code
**Status**: Not started  
**What to do**:

1. **Set up test simulation**:
   - Input: 1 neuron firing at 50 sp/s for 1 second
   - Create spike train: Poisson process with rate 50 Hz
   - Two output neurons: weights = 0.5 each, tau = [20 ms, 100 ms]

2. **Run simulation and record**:
   - Record membrane potentials and spikes
   - Print spike counts (first neuron should have 0 spikes)

3. **Analytic solution**:
   - For LIF neuron: `v(t) = I * tau * (1 - exp(-t/tau))`
   - With constant input current `I = rate * weight * dt` (approximate)
   - For tau=20ms, I=0.5: `v(t) = 0.5 * 0.02 * (1 - exp(-t/0.02))`
   - Maximum: `v_max = I * tau = 0.5 * 0.02 = 0.01` (below threshold=1.0, so no spikes)

4. **Plot comparison**:
   - Plot simulated `v(t)` vs analytic `v(t)`
   - They should match closely

**Dependencies**: Task 3A

---

### **TASK 4: Evaluating Fit to Data** ⚠️

**Status**: Not started  
**What to do**:

1. **Write `evaluate_network()` function**:
   - **Inputs**: `net`, `test_range`, `length`, `batch_size`, `dt`
   - **Outputs**: `(test_loss, null_loss)`
   - Compute MSE between network output and target velocities
   - Null loss: MSE of all zeros vs target (baseline)

2. **Write visualization function**:
   - Plot spikes from hidden layers (raster plots)
   - Compute and plot firing rates (histograms)
   - Plot network outputs vs targets

3. **Initialize and test network**:
   - Create network: 1 hidden layer (100 spiking neurons) → 1 output layer (2 non-spiking neurons)
   - Run on random batch
   - Plot: input spikes, hidden layer spikes, output velocities, target velocities
   - Compute firing rates for hidden layer
   - **Goal**: Firing rates should be 20-100 Hz, outputs should be in reasonable range (±4)

4. **Print baseline loss**:
   - Print test loss and null loss for untrained network

**Key Points**:
- Use `torch.no_grad()` when evaluating (no gradients needed)
- Tune weight initialization to get reasonable firing rates
- If all neurons fire identically, try different weight initializations

**Dependencies**: Task 2B (batched_data), Task 3A (SNNLayer)

---

### **TASK 5: Training** ⚠️

**Status**: Not started  
**What to do**:

1. **Set up single-layer network** (non-spiking output only):
   - Input spikes → 2 non-spiking LIF neurons (output)
   - This is simpler and trains faster (2 minutes vs 30 minutes)

2. **Find good initialization**:
   - Tune weights and time constants so outputs are in right range
   - Outputs should be roughly ±2-4 (matching whitened data)

3. **Set up training loop**:
   - **Optimizer**: `torch.optim.Adam(net.parameters(), lr=0.001)`
   - **Loss**: MSE (already defined as `mse = nn.MSELoss()`)
   - **Epochs**: 10
   - **Batch size**: 32
   - **Batches per epoch**: 40

4. **Training loop**:
   ```python
   for epoch in range(num_epochs):
       for x, y in batched_data(train_range, ...):
           y_out = net(x)
           loss = mse(y_out, y)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           # Clamp tau to be positive (e.g., tau = torch.clamp(tau, min=1e-3))
   ```

5. **Plot loss curves**:
   - Training loss vs epoch
   - Validation loss vs epoch
   - Null loss (horizontal line)
   - Use `plt.semilogy()` for log scale

6. **Plot trained outputs**:
   - 8 random time windows
   - Compare network output (raw and smoothed) vs target
   - Use `savgol_filter()` for smoothing

7. **Print test loss**:
   - Evaluate on test set
   - Should be better than null loss (~1.04) but may not be perfect (~0.55 target)

**Key Points**:
- Keep `tau` positive: `net.tau.data = torch.clamp(net.tau.data, min=1e-3)`
- Monitor validation loss to avoid overfitting
- Save model if needed: `torch.save(net.state_dict(), 'model.pt')`

**Dependencies**: Task 4 (evaluation function), Task 2B (batched_data), Task 3A (SNNLayer)

**Template provided**: Yes (commented training loop)

---

### **TASK 6: Longer Length Decoding** ⚠️

**Status**: Not started  
**What to do**:

1. **Implement `decoding_plot()` function**:
   - **Inputs**: `net`, `dt_decoding=0.2`, `decoding_start=1000`, `decoding_length=15`, `length=1`, `dt=1e-3`
   - **Strategy**: 
     - Take 15-second segment of test data
     - Sample every 0.2 seconds (75 points)
     - For each point, take 1-second window BEFORE it
     - Run simulation for 1 second
     - Use **final timestep** of output as prediction
     - Compare to actual velocity at that time

2. **Create batches**:
   - Generate overlapping 1-second windows
   - Run network on all windows
   - Extract final timestep from each output

3. **Plot results**:
   - 8 different 15-second segments (4x2 grid)
   - Each segment shows x and y velocities
   - Plot actual (dashed) vs predicted (solid)
   - Should look like the `fits.png` image

**Key Points**:
- Using final timestep avoids initialization transients
- Overlapping windows allow smooth predictions
- This is the "real" decoding task (brain-machine interface)

**Dependencies**: Task 5 (trained network), Task 2B (data batching)

**Template provided**: Yes (commented function)

---

### **TASK 7: Comparing Spiking and Non-Spiking** ⚠️

**Status**: Not started  
**What to do**:

1. **Train network with spiking hidden layer**:
   - Architecture: Input → Hidden (spiking, 100 neurons) → Output (non-spiking, 2 neurons)
   - **Warning**: Training will be much slower (30 minutes vs 2 minutes)
   - Use same training procedure as Task 5

2. **Compare results**:
   - Test loss for spiking vs non-spiking
   - Plot decoding results for both
   - Analyze differences in performance

3. **Plot weight and time constant distributions**:
   - Histogram of weights for both models
   - Histogram of time constants for spiking model
   - Compare distributions

4. **Optional**: Compare to Perez et al. 2021 paper on time constant heterogeneity

**Key Points**:
- Spiking networks are more biologically realistic but slower to train
- Time constants should be distributed (heterogeneous)
- Weights may have different distributions for spiking vs non-spiking

**Dependencies**: Task 5 (training code), Task 6 (decoding plot)

---

## Recommended Workflow

### Phase 1: Data Exploration (Tasks 1-2)
1. Complete Task 1A (preprocessing and statistics)
2. Complete Task 1B (plotting)
3. Complete Task 2A (why not continuous ranges)
4. Complete Task 2B (batching) - **Critical for everything else**

### Phase 2: Model Development (Tasks 3-4)
1. Complete Task 3A (SNNLayer implementation)
2. Complete Task 3B (verification)
3. Complete Task 4 (evaluation and initialization)

### Phase 3: Training (Tasks 5-7)
1. Complete Task 5 (training single-layer non-spiking) - **Start here for quick results**
2. Complete Task 6 (longer decoding)
3. Complete Task 7 (spiking vs non-spiking comparison)

---

## Key Concepts to Understand

### 1. Leaky Integrate-and-Fire (LIF) Neuron
- Membrane potential: `dv/dt = (I - v) / tau`
- Discrete update: `v_new = v * exp(-dt/tau) + I * (1 - exp(-dt/tau))`
- Spikes when `v > threshold`, then reset to `reset_value`

### 2. Surrogate Gradient Descent
- Standard Heaviside function (step function) has zero gradient everywhere
- Surrogate gradient uses smooth approximation for backpropagation
- Allows training spiking networks with gradient descent

### 3. Heterogeneous Time Constants
- Each neuron can have different `tau` (time constant)
- Makes network more flexible and biologically realistic
- Time constants are trainable parameters

### 4. Data Whitening
- Normalize data to mean=0, std=1
- Helps with training stability
- Important for neural networks

### 5. Batching and Interpolation
- Convert spike times to binary arrays (binned)
- Interpolate velocities to match simulation time steps
- Use generator functions for efficient data loading

---

## Common Pitfalls

1. **Forgetting to whiten data**: Leads to poor training
2. **Wrong time constants**: Too small → too fast, too large → too slow
3. **Negative time constants**: Must clamp `tau > 0`
4. **Wrong batch shapes**: Double-check `(batch_size, num_neurons, num_time_points)`
5. **Not using `torch.no_grad()` for evaluation**: Wastes computation
6. **Poor initialization**: Leads to all neurons firing identically
7. **Forgetting to reset after spikes**: Membrane potential doesn't reset
8. **Wrong interpolation**: Velocities must match simulation time points

---

## Resources

- **SPyTorch Tutorial**: https://github.com/fzenke/spytorch
- **Glaser et al. 2020**: Machine learning for neural decoding
- **Past Coursework Sample**: Reference implementation style
- **Week 1-3 Tutorials**: LIF neurons, networks, data analysis

---

## Expected Results

- **Task 5**: Test loss ~0.5-0.6 (better than null loss ~1.04)
- **Task 6**: Decoding plots should show reasonable fit to data
- **Task 7**: Spiking network may perform similarly or slightly worse than non-spiking, but is more biologically realistic

---

## Next Steps

1. Start with Task 1A - get data loaded and preprocessed
2. Work through tasks sequentially
3. Test each component before moving to the next
4. Use templates and examples from tutorials
5. Ask for help if stuck on specific implementation details

Good luck! 🚀

