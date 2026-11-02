
# ==========================================
# NEUROSCAN AI - GRAD-CAM EXPLAINABILITY
# ==========================================

import torch
import torch.nn.functional as F


class GradCAM:
    """
    Generate Grad-CAM heatmaps for a CNN.
    """

    def __init__(self, model, target_layer):

        self.model = model
        self.activations = None

        # Capture feature maps during forward pass
        self.hook = target_layer.register_forward_hook(
            self._save_activations
        )

    def _save_activations(
        self, module, inputs, output
    ):
        self.activations = output

    def generate(
        self,
        image_tensor,
        class_idx=None
    ):
        """
        image_tensor shape: [1, 3, 224, 224]

        Returns:
            heatmap: NumPy array [224, 224]
            class_idx: Explained class index
        """

        self.model.eval()

        # Enable gradients for explainability
        image_tensor = (
            image_tensor.detach()
            .clone()
            .requires_grad_(True)
        )

        with torch.enable_grad():

            # Forward pass
            logits = self.model(image_tensor)

            # Explain predicted class by default
            if class_idx is None:
                class_idx = int(
                    logits.argmax(dim=1).item()
                )

            # Calculate gradient of target class
            gradients = torch.autograd.grad(
                outputs=logits[0, class_idx],
                inputs=self.activations
            )[0]

            # Global average pooling of gradients
            weights = gradients.mean(
                dim=(2, 3),
                keepdim=True
            )

            # Weighted combination of feature maps
            cam = (
                weights * self.activations
            ).sum(dim=1, keepdim=True)

            cam = F.relu(cam)

            # Resize heatmap to input image size
            cam = F.interpolate(
                cam,
                size=image_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            # Convert to a 2D CPU tensor
            cam = cam[0, 0].detach().cpu()

            # Normalize to [0, 1]
            cam = cam / cam.max().clamp(min=1e-8)

        return cam.numpy(), class_idx

    def close(self):
        # Remove forward hook when finished
        self.hook.remove()
