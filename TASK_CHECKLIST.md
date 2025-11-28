# Coursework Task Checklist

## ✅ Progress Tracking

### Task 1: Load and Plot the Data
- [X] **Task 1A**: Preprocess and compute basic statistics
  - [X] Whiten velocities (mean=0, std=1)
  - [X] Print number of neurons
  - [X] Print total number of spikes
  - [X] Print experiment duration
  - [X] Print/estimate spike sampling rate
  - [X] Print/estimate velocity sampling rate

- [X] **Task 1B**: Plot the data
  - [X] Raster plot (whole dataset)
  - [X] Raster plot (1000-1010 seconds)
  - [X] Velocity plot (whole dataset)
  - [X] Velocity plot (1000-1010 seconds)
  - [X] Firing rate bar chart
  - [X] Velocity trajectory in (x,y) space

### Task 2: Divide Data into Test/Train and Batches
- [X] **Task 2A**: Why not continuous time ranges?
  - [X] Compute speeds from velocities
  - [X] Plot speeds (with smoothing)
  - [X] Compute and print speed statistics
  - [X] Explain why continuous splitting is bad

- [X] **Task 2B**: Equally distributed segments
  - [X] Divide data into segments
  - [X] Assign segments to train/val/test sets
  - [X] Implement `batched_data()` generator function
  - [X] Convert spikes to binary arrays
  - [X] Interpolate velocities to simulation time points
  - [X] Plot sample batch (4 examples)

### Task 3: Spiking Neural Network Model
- [X] **Task 3A**: Single layer simulation code
  - [X] Create `SNNLayer` class (nn.Module)
  - [X] Implement weight matrix (trainable)
  - [X] Implement time constants (trainable, per-neuron)
  - [X] Implement forward pass (LIF dynamics)
  - [X] Support spiking and non-spiking output
  - [X] Initialize weights and time constants
  - [X] Ensure tau stays positive
  - [X] (Optional) Create multi-layer network class

- [X] **Task 3B**: Verify your code
  - [X] Set up test simulation (50 sp/s input, 2 neurons)
  - [X] Run simulation and record membrane potentials
  - [X] Plot membrane potentials and spikes
  - [X] Print spike counts (first neuron should have 0)
  - [X] Derive analytic solution
  - [X] Plot simulated vs analytic solution

### Task 4: Evaluating Fit to Data
- [X] **Task 4**: Evaluation and initialization
  - [X] Write `evaluate_network()` function
  - [X] Compute test loss (MSE)
  - [X] Compute null loss (baseline)
  - [X] Write visualization function for network internals
  - [X] Create network (100 spiking → 2 non-spiking)
  - [X] Plot input spikes
  - [X] Plot hidden layer spikes
  - [X] Plot output vs target velocities
  - [X] Compute and plot firing rates (histogram)
  - [X] Tune initialization to get reasonable firing rates (20-100 Hz)
  - [X] Print baseline loss

### Task 5: Training
- [X] **Task 5**: Train single-layer network
  - [X] Create single-layer network (non-spiking output)
  - [X] Find good initialization
  - [X] Set up optimizer (Adam, lr=0.001)
  - [X] Implement training loop
  - [X] Clamp tau to be positive
  - [X] Track training and validation loss
  - [X] Plot loss curves (log scale)
  - [X] Evaluate on test set
  - [X] Print test loss and null loss
  - [X] Plot trained outputs (8 random windows)
  - [X] Plot smoothed outputs

### Task 6: Longer Length Decoding
- [X] **Task 6**: Decoding plot
  - [X] Implement `decoding_plot()` function
  - [X] Create overlapping 1-second windows
  - [X] Sample every 0.2 seconds
  - [X] Use final timestep as prediction
  - [X] Plot 8 different 15-second segments
  - [X] Compare actual vs predicted velocities
  - [X] Results should match `fits.png` style

### Task 7: Comparing Spiking and Non-Spiking
- [X] **Task 7**: Spiking vs non-spiking comparison
  - [X] Train network with spiking hidden layer
  - [X] Compare test losses
  - [X] Plot decoding results for both
  - [X] Plot weight distributions (both models)
  - [X] Plot time constant distribution (spiking model)
  - [X] (Optional) Compare to Perez et al. 2021

---

## Quick Reference

### Key Functions to Implement
1. `batched_data(range_to_use, dt, length, batch_size)` - Data generator
2. `SNNLayer(n_in, n_out, spiking)` - Single layer of LIF neurons
3. `evaluate_network(net, test_range, ...)` - Evaluation function
4. `decoding_plot(net, ...)` - Long-term decoding visualization

### Key Parameters
- `dt = 1e-3` (1 ms simulation time step)
- `length = 1` (1 second batches)
- `batch_size = 32` (training), `64` (default)
- Time constants: 20-100 ms (spiking), 200-1000 ms (non-spiking)
- Learning rate: `0.001`
- Epochs: `10`
- Batches per epoch: `40`

### Key Formulas
- **LIF update**: `v_new = v * exp(-dt/tau) + I * (1 - exp(-dt/tau))`
- **Whitening**: `x_white = (x - mean(x)) / std(x)`
- **Speed**: `speed = sqrt(vx^2 + vy^2)`
- **Firing rate**: `rate = num_spikes / duration`

---

## Notes
- Start with Task 1A and work sequentially
- Test each component before moving to the next
- Use `torch.no_grad()` for evaluation
- Keep `tau > 0` (clamp if needed)
- Whitening is important for training
- Good initialization is crucial

---

## Estimated Time
- Task 1: 2-3 hours
- Task 2: 3-4 hours
- Task 3: 4-6 hours
- Task 4: 2-3 hours
- Task 5: 2-3 hours (training time: ~2 minutes)
- Task 6: 1-2 hours
- Task 7: 3-4 hours (training time: ~30 minutes)

**Total**: ~20-30 hours of work

