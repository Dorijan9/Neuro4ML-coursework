# Coursework Task Checklist

## ✅ Progress Tracking

### Task 1: Load and Plot the Data
- [ ] **Task 1A**: Preprocess and compute basic statistics
  - [ ] Whiten velocities (mean=0, std=1)
  - [ ] Print number of neurons
  - [ ] Print total number of spikes
  - [ ] Print experiment duration
  - [ ] Print/estimate spike sampling rate
  - [ ] Print/estimate velocity sampling rate

- [ ] **Task 1B**: Plot the data
  - [ ] Raster plot (whole dataset)
  - [ ] Raster plot (1000-1010 seconds)
  - [ ] Velocity plot (whole dataset)
  - [ ] Velocity plot (1000-1010 seconds)
  - [ ] Firing rate bar chart
  - [ ] Velocity trajectory in (x,y) space

### Task 2: Divide Data into Test/Train and Batches
- [ ] **Task 2A**: Why not continuous time ranges?
  - [ ] Compute speeds from velocities
  - [ ] Plot speeds (with smoothing)
  - [ ] Compute and print speed statistics
  - [ ] Explain why continuous splitting is bad

- [ ] **Task 2B**: Equally distributed segments
  - [ ] Divide data into segments
  - [ ] Assign segments to train/val/test sets
  - [ ] Implement `batched_data()` generator function
  - [ ] Convert spikes to binary arrays
  - [ ] Interpolate velocities to simulation time points
  - [ ] Plot sample batch (4 examples)

### Task 3: Spiking Neural Network Model
- [ ] **Task 3A**: Single layer simulation code
  - [ ] Create `SNNLayer` class (nn.Module)
  - [ ] Implement weight matrix (trainable)
  - [ ] Implement time constants (trainable, per-neuron)
  - [ ] Implement forward pass (LIF dynamics)
  - [ ] Support spiking and non-spiking output
  - [ ] Initialize weights and time constants
  - [ ] Ensure tau stays positive
  - [ ] (Optional) Create multi-layer network class

- [ ] **Task 3B**: Verify your code
  - [ ] Set up test simulation (50 sp/s input, 2 neurons)
  - [ ] Run simulation and record membrane potentials
  - [ ] Plot membrane potentials and spikes
  - [ ] Print spike counts (first neuron should have 0)
  - [ ] Derive analytic solution
  - [ ] Plot simulated vs analytic solution

### Task 4: Evaluating Fit to Data
- [ ] **Task 4**: Evaluation and initialization
  - [ ] Write `evaluate_network()` function
  - [ ] Compute test loss (MSE)
  - [ ] Compute null loss (baseline)
  - [ ] Write visualization function for network internals
  - [ ] Create network (100 spiking → 2 non-spiking)
  - [ ] Plot input spikes
  - [ ] Plot hidden layer spikes
  - [ ] Plot output vs target velocities
  - [ ] Compute and plot firing rates (histogram)
  - [ ] Tune initialization to get reasonable firing rates (20-100 Hz)
  - [ ] Print baseline loss

### Task 5: Training
- [ ] **Task 5**: Train single-layer network
  - [ ] Create single-layer network (non-spiking output)
  - [ ] Find good initialization
  - [ ] Set up optimizer (Adam, lr=0.001)
  - [ ] Implement training loop
  - [ ] Clamp tau to be positive
  - [ ] Track training and validation loss
  - [ ] Plot loss curves (log scale)
  - [ ] Evaluate on test set
  - [ ] Print test loss and null loss
  - [ ] Plot trained outputs (8 random windows)
  - [ ] Plot smoothed outputs

### Task 6: Longer Length Decoding
- [ ] **Task 6**: Decoding plot
  - [ ] Implement `decoding_plot()` function
  - [ ] Create overlapping 1-second windows
  - [ ] Sample every 0.2 seconds
  - [ ] Use final timestep as prediction
  - [ ] Plot 8 different 15-second segments
  - [ ] Compare actual vs predicted velocities
  - [ ] Results should match `fits.png` style

### Task 7: Comparing Spiking and Non-Spiking
- [ ] **Task 7**: Spiking vs non-spiking comparison
  - [ ] Train network with spiking hidden layer
  - [ ] Compare test losses
  - [ ] Plot decoding results for both
  - [ ] Plot weight distributions (both models)
  - [ ] Plot time constant distribution (spiking model)
  - [ ] (Optional) Compare to Perez et al. 2021

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

