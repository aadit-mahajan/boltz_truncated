# Boltz-2 latent-reuse fork

This fork separates structure sampling from affinity scoring.

The structure pass writes `structure_cache_<record>.npz` containing the trunk
single/pair states (`s`, `z`) and the selected denoised coordinates. An
affinity-only `predict_step` can pass these tensors as `cached_structure`; the
model then skips the diffusion sampler and feeds the cached pose to the
affinity head.

The cache must be cropped with the same `AffinityCropper` token/atom mapping
before it is supplied to the model. The current production loader is left
unchanged until that mapping is wired, so existing runs retain their original
behavior.
