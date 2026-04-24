import torch
import torch.nn.functional as F
import cv2
import numpy as np

class YOLOExplain:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=0):
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # For YOLO, we usually take the max score across the grid
        # Output is [B, 8400, NC + 4] or similar
        # We'll backprop from the class score
        # Note: This is a simplified Grad-CAM for detection
        score = output[0, :, 4 + class_idx].max()
        score.backward()

        # Weighted combination of channels
        gradients = self.gradients
        activations = self.activations
        
        # Pool the gradients across the spatial dimensions
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Compute weighted sum of activations
        cam = torch.sum(weights * activations, dim=1).squeeze()
        
        # ReLU and normalization
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.detach().cpu().numpy()

def apply_heatmap_to_image(image, mask):
    """
    Overlay heatmap on original BGR image.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Composite
    result = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return result
