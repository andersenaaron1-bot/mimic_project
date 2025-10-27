import torch

def patient_block_mask(pid: torch.Tensor, is_static: torch.Tensor | None = None):
    #[B,T,T] True=keep, False=mask. Within-patient attention (+static tokens global evt)
    B, T = pid.shape
    same = pid.unsqueeze(-1).expand(B,T,T).eq(pid.unsqueeze(-2).expand(B,T,T))
    if is_static is not None:
        static_row = is_static.unsqueeze(-1).expand(B,T,T)
        static_col = is_static.unsqueeze(-2).expand(B,T,T)
        same = same | static_col | static_row
    return same
