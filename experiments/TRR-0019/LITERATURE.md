# Primary implementation references

- PyTorch CUDA semantics, https://docs.pytorch.org/docs/2.10/notes/cuda.html — CUDA graph input/output storage, replay semantics and sharing temporary memory across captures replayed in order. The local installed runtime isPyTorch 2.10.0+cu128; local implementation was inspected in addition to the current official documentation.
- PyTorch CUDA graph overview, https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/ — reducing CPU launch overhead while reusing operations at stable addresses. This motivates execution capture, not a claim of accuracy or measured speedup in our setting.
- Installed Transformers 5.3.0 LlamaAttention.forward and integrations/sdpa_attention.py were inspected directly. The new kernel implements single-query grouped-query attention over the shared past plus candidate-local current K/V. It deliberately records numerical differences from native SDPA and is not claimed bitwise equivalent.

All numerical results and speed claims come from the task's own experiment receipts. No external benchmark is substituted for these measurements.
