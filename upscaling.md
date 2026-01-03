If the denoiser was trained on latents whose joint channel statistics look like some distribution z \sim (\mu,\Sigma), then a naive per-channel bilinear/bicubic upscale is doing two slightly-wrong things:
	1.	it treats channels as independent scalars (ignores \Sigma_{ij}, i\neq j), and
	2.	it effectively changes the metric in latent space (channels with bigger variance dominate interpolation and edge decisions), which can push you off the manifold the denoiser expects.

Here are concrete ways to make the spatial upscale “covariance-aware” and exploit intra-channel variance.

⸻

1) Whiten → upscale → re-color (global or local)

This is the simplest “use covariance” trick, and it’s surprisingly effective.

Let each pixel be a 16-D vector z_p \in \mathbb{R}^{16}. Estimate dataset (or batch) mean/cov:

\mu = \mathbb{E}[z_p], \quad \Sigma = \mathbb{E}[(z_p-\mu)(z_p-\mu)^T]

Compute a whitening transform. With Cholesky \Sigma = LL^T:
	•	Whiten: y_p = L^{-1}(z_p-\mu)  (now channels are ~unit variance, decorrelated)
	•	Upscale spatially: \hat y = \text{Upsample}(y) (bilinear/bicubic/Lanczos/whatever)
	•	Re-color: \hat z_p = \mu + L \hat y_p

Intuition: you’re doing interpolation in a space where Euclidean distance corresponds to Mahalanobis distance in the original latent:
\|y_a-y_b\|_2^2 = (z_a-z_b)^T \Sigma^{-1} (z_a-z_b)
So cross-channel covariance directly shapes “what it means” for two latent vectors to be similar during interpolation.

Local variant (better, riskier): estimate \Sigma_p per patch (e.g., 7\times7 neighborhood), but stabilize it with shrinkage:
\Sigma'_p = (1-\alpha)\Sigma_p + \alpha \Sigma_{\text{global}} + \epsilon I
Then whiten/upscale/recolor either (a) per tile, blending overlaps, or (b) with a smoothly varying transform. This tends to preserve edges better because local channel correlations often encode structure.

⸻

2) PCA / eigenbasis upscale with variance-aware frequency handling

Compute eigendecomposition \Sigma = U\Lambda U^T. Project each pixel:

c_p = U^T(z_p - \mu)

Now you have “principal components” ordered by dataset variance \Lambda_{kk}.

Two useful tricks:
	•	Component-specific kernels: upscale high-variance components with sharper kernels (preserve detail), and low-variance components with smoother kernels (reduce ringing / noise injection). You can do this with a mild Laplacian-pyramid: push more high-frequency into the top PCs and less into the bottom PCs.
	•	Variance equalization: scale c_k by 1/\sqrt{\Lambda_{kk}} (this is whitening), upscale, then scale back.

This uses intra-channel (component) variance as a knob for how aggressively you preserve or smooth spatial frequencies.

⸻

3) Vector-valued edge-aware upsampling using a Mahalanobis metric

Treat the latent as a 16-channel vector image and run an edge-aware filter / upsampler where “range distance” is covariance-weighted.

For example, a (joint) bilateral/guided upsampling weight between pixels p,q:

w(p,q) = \exp\!\left(-\frac{\|p-q\|^2}{2\sigma_s^2}\right)\;
\exp\!\left(-\frac{(z_p-z_q)^T\Sigma^{-1}(z_p-z_q)}{2\sigma_r^2}\right)

Then compute the upsampled value as a normalized weighted sum over a neighborhood in the low-res grid mapped to the high-res pixel.

This couples channels automatically: an “edge” is where the latent vector changes under the correct metric, not where a single channel jumps. If you use local \Sigma_p (shrinked), it becomes an adaptive vector-edge detector.

⸻

4) Moment-preserving upscaling (match \mu,\Sigma before denoising)

Even after a smart upscale, you can drift in per-channel variance/covariance (especially with aggressive kernels).

A cheap fix: after upsampling, apply a small affine correction so that the set of pixel vectors in \hat z matches target mean/covariance (global or patchwise). Think of it as “color transfer” but in 16-D latent space.

If \hat \mu, \hat \Sigma are the upsampled stats, map:
z^{*}_p = \mu + L \hat L^{-1} (\hat z_p - \hat \mu)
where L L^T=\Sigma and \hat L \hat L^T = \hat \Sigma.

This doesn’t invent detail, but it does keep the denoiser from seeing weird channel-energy ratios it wasn’t trained on.

⸻

5) Make the denoiser do the hard part: covariance-weighted “data fidelity” during refinement

If you can afford a few denoising steps, treat upscaling as an inverse problem:
	•	Unknown: high-res latent z_{HR}
	•	Known: low-res latent z_{LR}
	•	Constraint: downsample D(z_{HR}) \approx z_{LR}

Solve a MAP-style objective:
\min_{z_{HR}} \;\; \|D(z_{HR}) - z_{LR}\|_{\Sigma^{-1}}^2 \;+\; \lambda \cdot \text{Prior}(z_{HR})
where the prior is implicitly given by the diffusion/denoiser (via standard diffusion posterior sampling / guidance ideas).

Key point: the covariance-weighted norm is exactly where your cross-channel stats belong—channels that are strongly correlated or low-variance are handled in a way consistent with training distribution.

A practical recipe:
	1.	initialize z_{HR} with “whiten → upscale → recolor”
	2.	run N denoise steps with a guidance term proportional to \nabla \|D(z)-z_{LR}\|_{\Sigma^{-1}}^2

This often beats any purely interpolative method because it reintroduces plausible high-frequency structure on-manifold.

⸻

What to watch out for
	•	Variance imbalance isn’t automatically “unbalanced training.” Some VAE latents intentionally allocate energy unevenly. The real issue is whether your upscaler distorts the joint statistics the denoiser expects.
	•	Local covariance can be noisy. Always shrink/blend with a global \Sigma and add \epsilon I. Otherwise you get unstable transforms and shimmering.
	•	Per-channel rescaling without covariance can hurt. If you only normalize by channel stddev (diagonal) you may still break meaningful correlations.

⸻

If you want one “best bang for buck” experiment
	1.	compute global \mu,\Sigma from a decent batch of latents
	2.	do whiten → upscale → recolor
	3.	optionally add the moment-preserving correction after upscaling
	4.	compare decoded images before any diffusion refinement, then compare after a small fixed number of denoise steps (so you can see which method gives the denoiser a better starting point)

If you tell me what upscale factor you care about (2×, 4×) and whether you can run any diffusion refinement steps, I can propose a very specific algorithm (including how to do stable local-\Sigma tiling) that’s tuned for “latent SR” rather than pixel SR.
