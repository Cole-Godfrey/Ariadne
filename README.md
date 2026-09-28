# GradForge

A neural-network framework written with Python and NumPy. It includes a tensor computation graph, reverse-mode automatic differentiation, layers, optimizers, and a reproducible multilayer perceptron experiment on [MNIST](https://yann.lecun.org/exdb/mnist/).

The default experiment reached **97.91% accuracy on the 10,000-image MNIST test set** (seed 42, validation-selected epoch 17 of 20). The official 60,000-image training set is split into 55,000 training and 5,000 validation images. The test set is used once, after selecting the best checkpoint by validation accuracy.

## Reproduce

With Python 3.10–3.13 and internet access, clone the repository and run this one command from its root:

```bash
./reproduce.sh
```

The script creates `.venv`, installs pinned NumPy, runs finite-difference gradient tests, downloads and checksum-verifies MNIST, trains the MLP, and writes `results/history.csv`, `results/curves.svg`, `results/summary.json`, and `results/best_model.npz`. The first run downloads about 12 MB of compressed data. To rerun training without reinstalling dependencies, use `.venv/bin/python -m gradforge.experiment`. Run `--help` for optional settings.

The defaults are fixed: seed 42, layers `784 → 256 → 128 → 10`, ReLU hidden activations, He-normal weights, zero biases, Adam with learning rate 0.001, batch size 128, and 20 epochs. Pixels are converted to `float32` in `[0, 1]`. `numpy.random.SeedSequence` derives separate generators for the train/validation split, weight initialization, and epoch shuffling. Reproductions on different BLAS implementations may vary slightly because floating-point matrix multiplication can differ.

![Training and validation loss and accuracy](results/curves.svg)

## How backpropagation works here

Each `Tensor` stores a NumPy array, a gradient slot, its parent tensors, and a small backward function. Operations such as matrix multiplication, addition, multiplication, reduction, and ReLU create a new tensor and record the local gradient rule. The forward pass builds a directed acyclic computation graph. Starting at a scalar loss, `backward()` visits every reachable node in topological order, then applies local rules in reverse order. If several paths lead to the same tensor, their gradient contributions are added.

For a linear layer, `Y = XW + b`, an upstream gradient `G = ∂L/∂Y` gives `∂L/∂X = GWᵀ`, `∂L/∂W = XᵀG`, and `∂L/∂b = ΣᵢGᵢ`. The bias is broadcast across the batch on the forward pass, so its backward rule sums over the batch. More generally, `_unbroadcast` reduces gradients along dimensions expanded by NumPy broadcasting. ReLU sends the upstream gradient through entries where its input is positive and zeroes it elsewhere.

`Tensor.softmax()` uses a shifted exponential for stability. Its backward rule computes the Jacobian-vector product directly: for probabilities `p` and upstream gradient `g`, the input gradient is `p ⊙ (g − Σ(g ⊙ p))`. Training uses a fused softmax/cross-entropy function. For logits `z` and target class `y`, it computes `log(Σⱼ exp(zⱼ − max(z))) − (zᵧ − max(z))`, averaged over the batch. Its gradient with respect to each logit is `(p − one_hot(y)) / batch_size`. The shift prevents large logits from overflowing.

`SGD` applies `θ ← θ − ηg`. `Adam` keeps exponential moving averages of the gradient and squared gradient, corrects their initialization bias, and applies `θ ← θ − ηm̂/(√v̂ + ε)`. The model is a sequence of `Linear` and `ReLU` layers. It is trained entirely through the tensor graph; NumPy only supplies array storage and arithmetic.

## Verification and results

`python -m unittest discover -s tests -v` checks gradients against centered finite differences for broadcasting, matrix multiplication, ReLU with shared graph paths, softmax, stable cross-entropy, and every parameter of a small MLP. It also checks SGD and Adam updates. The default run's measured results are in [summary.json](results/summary.json) and the per-epoch values are in [history.csv](results/history.csv).

MNIST is downloaded from the [CVDF mirror](https://github.com/cvdfoundation/mnist) or a secondary mirror and checked against the known MD5 digests before parsing the IDX headers. The data and saved weights are ignored by Git; the small metric and plot artifacts are committed.
