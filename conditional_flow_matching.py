import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_matching.utils import ModelWrapper
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.path.scheduler import CondOTScheduler

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return torch.sigmoid(x) * x

def get_activation(s_act):
    if s_act == "relu":
        return nn.ReLU(inplace=True)
    elif s_act == "sigmoid":
        return nn.Sigmoid()
    elif s_act == "softplus":
        return nn.Softplus()
    elif s_act == "linear":
        return None
    elif s_act == "tanh":
        return nn.Tanh()
    elif s_act == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == "softmax":
        return nn.Softmax(dim=1)
    elif s_act == "selu":
        return nn.SELU()
    elif s_act == "elu":
        return nn.ELU()
    elif s_act == "gelu":
        return nn.GELU()
    elif s_act == 'swish':
        return Swish()
    else:
        raise ValueError(f"Unexpected activation: {s_act}")

class ConditionalFlowMatching(nn.Module):
    def __init__(self, input_dim, cond_dim=None, hidden_dim=[256, 512, 256], dropout_prob=0.1):
        """
        Unified Flow Matching implementation with conditional support and classifier-free guidance
        
        Args:
            input_dim: Dimension of input data
            cond_dim: Dimension of conditioning information (None for unconditional)
            hidden_dim: Hidden layer dimension (default: 128)
            dropout_prob: Probability of dropping condition during training (for classifier-free guidance)
        """
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.dropout_prob = dropout_prob
        
        # Calculate network input dimension
        total_input_dim = input_dim + 1  # For data + time
        
        # Add conditioning dimension if specified
        if cond_dim is not None:
            total_input_dim += cond_dim
        
        # Define network architecture
        layers = []
        in_dim = total_input_dim
        for i, out_dim in enumerate(hidden_dim):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(hidden_dim) - 1:  # No activation after last layer
                layers.append(get_activation('swish'))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, input_dim))  # Final output layer
        
        self.net = nn.Sequential(*layers)

        # instantiate an affine path object
        self.path = AffineProbPath(scheduler=CondOTScheduler())
    
    def forward(self, x, t, cond=None):
        """
        Forward pass of the model
        
        Args:
            x: Input data [batch_size, input_dim]
            t: Time step [batch_size, 1]
            cond: Conditioning information [batch_size, cond_dim] or None
            
        Returns:
            Predicted velocity field [batch_size, input_dim]
        """
        # Concatenate input, time, and conditioning
        inputs = [x, t.reshape(-1, 1).float()]  # Ensure t has shape [batch_size, 1]
        
        if cond is not None:
            inputs.append(cond)
        
        model_input = torch.cat(inputs, dim=1)
        return self.net(model_input)
    
    def compute_loss(self, x1, cond=None):
        """
        Compute Flow Matching loss with optional conditioning
        
        Args:
            x1: Data samples from target distribution [batch_size, input_dim]
            cond: Conditioning information [batch_size, cond_dim] or None
            
        Returns:
            loss: Flow Matching loss (scalar tensor)
            x_t: Interpolated points [batch_size, input_dim]
            v_pred: Predicted velocity field [batch_size, input_dim]
            v_target: Target velocity field [batch_size, input_dim]
        """
        batch_size = x1.shape[0]
        
        # Sample random time steps uniformly in [0,1]
        t = torch.rand(batch_size, device=x1.device)
        
        # Sample from noise distribution (standard Gaussian)
        x0 = torch.randn_like(x1)

        path_sample = self.path.sample(t=t, x_0=x0, x_1=x1)
        
        # Apply classifier-free guidance: randomly drop conditions during training
        if cond is not None and self.dropout_prob > 0:
            # Create mask for condition dropout
            mask = torch.rand(batch_size, device=x1.device) < self.dropout_prob
            cond_drop = cond.clone()
            cond_drop[mask] = 0.0  # Drop out conditions
            dx_t_pred = self.forward(path_sample.x_t, t, cond=cond_drop)
        else:
            dx_t_pred = self.forward(path_sample.x_t, t, cond=cond)
        
        # Calculate loss
        loss = F.mse_loss(dx_t_pred, path_sample.dx_t)
        
        return loss, path_sample.x_t, dx_t_pred, path_sample.dx_t
    
    @torch.no_grad()
    def sample(self, num_samples=None, cond=None, steps=100, device='cpu', guidance_scale=1.5):
        """
        Generate samples using the trained model with optional classifier-free guidance
        
        Args:
            num_samples: Number of samples for unconditional generation
            cond: Conditioning information for conditional generation [batch_size, cond_dim]
            steps: Number of integration steps (default: 100)
            device: Device to use ('cpu' or 'cuda')
            guidance_scale: Strength of classifier-free guidance (0 = unconditional, >0 = conditional)
            
        Returns:
            Generated samples [num_samples or batch_size, input_dim]
        """
        self.eval()
        
        # Handle input combinations
        if cond is not None:
            batch_size = cond.shape[0]
            if num_samples is not None and num_samples != batch_size:
                print("Warning: num_samples ignored when cond is provided")
        elif num_samples is None:
            raise ValueError("Must provide either num_samples or cond")
        else:
            batch_size = num_samples
        
        # Start from Gaussian noise
        x = torch.randn(batch_size, self.input_dim, device=device)
        
        # Create null condition tensor
        null_cond = torch.zeros(batch_size, self.cond_dim, device=device) if self.cond_dim else None
        
        # Time steps from 1 to 0 (reverse process)
        timesteps = torch.linspace(1, 0, steps, device=device)
        
        # Euler integration with classifier-free guidance
        for i in range(steps - 1):
            t = timesteps[i].expand(batch_size)
            dt = timesteps[i] - timesteps[i+1]
            
            # No guidance for unconditional generation
            if guidance_scale <= 0 or cond is None:
                v = self.forward(x, t, cond=null_cond)
            # Apply classifier-free guidance
            else:
                # Predict with null condition (unconditional)
                v_uncond = self.forward(x, t, cond=null_cond)
                
                # Predict with actual condition (conditional)
                v_cond = self.forward(x, t, cond=cond)
                
                # Blend predictions using guidance scale
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            
            # Update x
            x = x + v * dt
        
        return x
