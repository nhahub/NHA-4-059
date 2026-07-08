"""
gradcam.py
==========

Grad-CAM for the CLIP-RN50 + LogisticRegression pipeline, with the
targeting bug from the original notebook fixed.

THE BUG (Section 5 of CH_Detection_Pipeline.ipynb)
---------------------------------------------------
The original code did:

    features = model.encode_image(img_tensor)
    target = features[0, features[0].argmax()]
    target.backward(...)

`features` is the raw 1024-dim CLIP embedding — not a class score. Picking
`argmax()` just grabs whichever embedding dimension happens to have the
largest raw value *for that image*, which is an arbitrary, unlabeled
direction in feature space, completely disconnected from the actual
`LogisticRegression` classifier that produces the reported predictions.
Two consequences: (1) the heatmap doesn't explain "why did the model call
this a garbage truck," so a heatmap that misses the logo doesn't tell you
anything about Clever-Hans reliance one way or the other, and (2) a
before/after logo comparison can silently be comparing two *different*
embedding dimensions (since adding the logo changes which dimension is
largest), making the pair not even comparable.

THE FIX
-------
Target the classifier's actual decision function for the predicted class,
i.e. the same linear score that `clf.predict()` uses:

    logit_c(x) = w_c . encode_image(x) + b_c

`w_c`/`b_c` come straight from the trained `LogisticRegression` (`coef_`,
`intercept_`), turned into a constant torch tensor. Backpropagating from
`logit_c` (not from an arbitrary embedding coordinate) makes the resulting
CAM answer "which pixels drove the actual predicted-class score" — the
thing the report/dashboard claims it's showing.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Overlay a Grad-CAM map (values in [0, 1], shape (H, W)) on `image`
    using a jet colormap. Returns an RGB uint8 array, shape (H, W, 3)."""
    h, w = cam.shape
    img_np = np.array(image.convert("RGB").resize((w, h)))

    heatmap_u8 = (cam * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap_color + (1 - alpha) * img_np).astype(np.uint8)
    return overlay


def attention_alignment_score(cam: np.ndarray, boxes) -> float:
    """Fraction of the CAM's total activation energy that falls inside the
    given (x1, y1, x2, y2) box(es), in the CAM's own pixel grid. This is the
    FR-5 "Attention Alignment Score" — >0.30 flags a spurious-region focus.

    `boxes` may be a single (x1, y1, x2, y2) tuple or an iterable of them
    (e.g. `ImageModifier.logo_regions(img)`, which returns two boxes).
    """
    if len(boxes) == 4 and all(isinstance(v, (int, float)) for v in boxes):
        boxes = [boxes]

    h, w = cam.shape
    total = cam.sum()
    if total <= 0:
        return 0.0

    mask = np.zeros_like(cam, dtype=bool)
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        mask[y1:y2, x1:x2] = True

    return float(cam[mask].sum() / total)


class GradCAM:
    def __init__(self, clip_model, target_layer_name: Optional[str] = None):
        """
        Parameters
        ----------
        clip_model:
            An instance of src.models.clip_model.CLIPModel with a
            classifier already loaded (`clip_model.classifier`).
        target_layer_name:
            Name of the module (within `model.visual`) to hook for Grad-CAM.
            If None, uses the *block containing* the last Conv2d layer in the
            visual backbone (see `_find_last_conv_name` for why it's the
            block, not the bare conv).
        """
        self.clip_model = clip_model
        self.model = clip_model.model

        if target_layer_name is None:
            target_layer_name = self._find_last_conv_name()
        self.target_layer_name = target_layer_name

        self.gradients = {}
        self.activations = {}
        self._fwd_handle = None
        self._bwd_handle = None
        self._register_hooks()

    def _find_last_conv_name(self) -> str:
        """Find the residual block *containing* the last Conv2d in the
        visual backbone, not the bare Conv2d itself.

        CLIP RN50's Bottleneck block does
        `relu3(bn3(conv3(x)) + identity)` — hooking the inner `conv3`
        submodule directly captures its raw output *before* batch-norm,
        before the residual add, and before the final ReLU. Raw pre-BN
        conv output has arbitrary per-channel scale (BN normally corrects
        that), and completely skips the skip-connection that carries most
        of the spatial signal forward in a ResNet. In practice this made
        Grad-CAM's channel-weighted sum dominated by whichever channel had
        the largest raw (unnormalized) magnitude, which zero-padding biases
        toward the image border/corner — producing the same corner hot-spot
        regardless of image content. Hooking the block's own output instead
        gives the real post-BN/post-residual/post-ReLU feature map.
        """
        last_conv_name = None
        for name, module in self.model.visual.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv_name = name
        if last_conv_name is None:
            raise RuntimeError("No Conv2d layer found in model.visual")

        if "." in last_conv_name:
            return last_conv_name.rsplit(".", 1)[0]
        return last_conv_name

    def _register_hooks(self) -> None:
        target_module = dict(self.model.visual.named_modules())[self.target_layer_name]

        def save_activation(module, inp, output):
            self.activations[self.target_layer_name] = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self.gradients[self.target_layer_name] = grad_output[0].detach()

        self._fwd_handle = target_module.register_forward_hook(save_activation)
        self._bwd_handle = target_module.register_full_backward_hook(save_gradient)

    def remove_hooks(self) -> None:
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
        if self._bwd_handle is not None:
            self._bwd_handle.remove()

    def _target_logit(self, features: torch.Tensor, class_index_in_clf: Optional[int]) -> Tuple[torch.Tensor, int]:
        """Build the classifier decision-function score to backprop from.

        If `class_index_in_clf` is None, uses the classifier's own
        argmax prediction for this image (i.e. explains the model's
        actual predicted class, not an arbitrary embedding coordinate).
        """
        if self.clip_model.classifier is None:
            raise RuntimeError(
                "GradCAM needs a loaded classifier to target its decision "
                "function. Call clip_model.load_classifier(path) first."
            )

        coef = torch.tensor(
            self.clip_model.classifier.coef_, dtype=features.dtype, device=features.device
        )  # (n_classes, 1024)
        intercept = torch.tensor(
            self.clip_model.classifier.intercept_, dtype=features.dtype, device=features.device
        )  # (n_classes,)

        logits = features @ coef.T + intercept  # (1, n_classes)

        if class_index_in_clf is None:
            class_index_in_clf = int(torch.argmax(logits[0]).item())

        return logits[0, class_index_in_clf], class_index_in_clf

    def __call__(self, img: Image.Image, class_index_in_clf: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Compute the Grad-CAM heatmap for `img`.

        Returns
        -------
        cam: np.ndarray, shape (224, 224), values in [0, 1]
        predicted_class_index: the classifier-internal class index the CAM
            was computed for (useful for labeling plots correctly).
        """
        img_tensor = self.clip_model.preprocess(img).unsqueeze(0).to(self.clip_model.device)
        img_tensor.requires_grad_(True)

        features = self.model.encode_image(img_tensor)
        self.model.zero_grad()

        target, class_index_in_clf = self._target_logit(features, class_index_in_clf)
        target.backward(retain_graph=True)

        grads = self.gradients[self.target_layer_name]
        acts = self.activations[self.target_layer_name]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))
        return np.array(cam_img) / 255.0, class_index_in_clf

    def overlay_heatmap(self, image: Image.Image, cam: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """Instance-method form of the module-level `overlay_heatmap`, matching
        the `Explainer.overlay_heatmap` contract in the SAD's class diagram."""
        return overlay_heatmap(image, cam, alpha=alpha)
