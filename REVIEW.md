# Review

## Summary
- Repository provides a set of example notebooks for matplotlib plotting, population visualization, and a CIFAR-10 image classification tutorial.
- Overall structure is clear, but the image classification notebook has a few correctness and reproducibility issues that should be addressed before relying on it as a tutorial.

## Findings
1. **Inference cell reuses the training dataloader instead of the test split.** The notebook comment claims to "Pick random photos from training set," yet the code later tries to fall back to `testloader` by checking `if dataiter == None`. Because `dataiter` is an iterator object, that conditional never becomes `True`, so the visualization always uses training batches and never exercises the held-out set.【F:notebooks/image-classifier.ipynb†L226-L235】
   - Fix: explicitly construct a fresh iterator from `testloader` (for example, `images, labels = next(iter(testloader))`) when you want to visualize generalization, or otherwise document that the preview intentionally shows training samples.

2. **Model stays in training mode during evaluation.** Both the qualitative preview and the quantitative accuracy calculation call `outputs = net(images)` without switching the network to evaluation mode. While this example network happens not to include dropout/batchnorm, the pattern is fragile and will silently break once such layers are added.【F:notebooks/image-classifier.ipynb†L231-L258】【F:notebooks/image-classifier.ipynb†L315-L328】
   - Fix: add `net.eval()` before running inference and wrap forward passes in `with torch.no_grad():` blocks to avoid gradient tracking and to ensure deterministic behavior.

3. **CPU/GPU device handling is omitted.** Every tensor and the model remain on the default CPU device, which will make the training cell extremely slow (and may even exhaust memory) when run inside the default Codespaces container. Providing optional CUDA/MPS support with a device helper greatly improves the tutorial's usability for readers who have access to accelerators.【F:notebooks/image-classifier.ipynb†L62-L205】
   - Fix: detect `torch.cuda.is_available()` / `torch.backends.mps.is_available()` and move the model, data, and loss calculations to that device when possible.
