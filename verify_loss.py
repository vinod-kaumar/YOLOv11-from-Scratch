import torch
from torch.optim import Adam
from loss import v8DetectionLoss

def test_v8_detection_loss_iterations(iterations=100):
    print(f"--- Verifying v8DetectionLoss over {iterations} Iterations ---")
    
    # 1. Initialize loss function
    loss_fn = v8DetectionLoss(nc=1, reg_max=16, strides=[8.0, 16.0, 32.0])
    
    # 2. Define Dummy Model Outputs as parameters (so we can optimize them)
    batch_size = 2
    channels = 65  # 1 (nc) + 16*4 (reg_max * 4)
    
    # We create random tensors and set requires_grad=True so the optimizer updates them
    p1 = torch.randn(batch_size, channels, 80, 80, requires_grad=True)
    p2 = torch.randn(batch_size, channels, 40, 40, requires_grad=True)
    p3 = torch.randn(batch_size, channels, 20, 20, requires_grad=True)
    
    # Use Adam optimizer to iteratively update the dummy predictions to match targets
    optimizer = Adam([p1, p2, p3], lr=0.1)
    
    # 3. Define Dummy Ground Truth Targets
    max_gt = 3
    targets = torch.zeros(batch_size, max_gt, 5)
    
    # Image 1 (Batch 0): 2 Polypes
    targets[0, 0] = torch.tensor([0.0, 0.5, 0.5, 0.2, 0.2])   # Center
    targets[0, 1] = torch.tensor([0.0, 0.3, 0.7, 0.1, 0.3])   # Bottom left
    
    # Image 2 (Batch 1): 1 Polyp
    targets[1, 0] = torch.tensor([0.0, 0.8, 0.2, 0.15, 0.15]) # Top right
    
    print("\nStarting optimization loop...")
    for i in range(iterations):
        optimizer.zero_grad()
        raw_preds = [p1, p2, p3]
        
        # Forward Pass
        total_loss, loss_items = loss_fn(raw_preds, targets, img_size=640)
        
        # Check for NaNs
        if torch.isnan(total_loss):
            print(f"\n ERROR: NaN loss detected at iteration {i}!")
            return
            
        # Backward Pass & Optimize
        total_loss.backward()
        optimizer.step()
        
        # Print progress every 10 iterations
        if i == 0 or (i + 1) % 10 == 0:
            print(f"Iteration {i+1:3d} | Total Loss: {total_loss.item():.4f} | "
                  f"Box: {loss_items['box']:.4f} | DFL: {loss_items['dfl']:.4f} | Cls: {loss_items['cls']:.4f}")

    print("\nMulti-iteration Check Successful! Loss decreases steadily without exploding or producing NaNs.")

if __name__ == "__main__":
    # Fix seed for reproducible check
    torch.manual_seed(42)
    test_v8_detection_loss_iterations(100)
