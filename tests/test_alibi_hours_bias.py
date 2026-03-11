import torch
import torch.nn as nn


class _IdentityRoPE(nn.Module):
    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return x


def test_alibi_hours_bias_prefers_recent_in_time() -> None:
    from ehr_hier.transformer.encoder import AETCausalAttention

    attn = AETCausalAttention(
        d_model=2,
        num_heads=1,
        rope_module=_IdentityRoPE(),
        dropout=0.0,
        enable_alibi_hours_bias=True,
        alibi_hours_max=1e9,
    )

    with torch.no_grad():
        # Remove content-based attention: Q and K are zeros, so only the time-bias
        # determines attention weights.
        attn.q_proj.weight.zero_()
        attn.q_proj.bias.zero_()
        attn.k_proj.weight.zero_()
        attn.k_proj.bias.zero_()

        # Values flow through unchanged.
        attn.v_proj.weight.copy_(torch.eye(2))
        attn.v_proj.bias.zero_()
        attn.out_proj.weight.copy_(torch.eye(2))
        attn.out_proj.bias.zero_()

        # Make the bias strong so the test is crisp.
        assert attn._alibi_slopes_raw is not None
        raw = torch.log(torch.expm1(torch.tensor(10.0))).to(attn._alibi_slopes_raw)
        attn._alibi_slopes_raw.fill_(raw)

    # Three tokens with a huge jump in time to token 2.
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]])  # (B=1,S=3,D=2)
    times = torch.tensor([[0.0, 1.0, 100.0]], dtype=torch.float)  # hours
    mask = torch.ones((1, 3), dtype=torch.long)

    out = attn(x, times, mask)

    # At position 1 (t=1h), recent-in-time key is token 1 itself.
    assert out[0, 1, 1].item() > 0.8

    # At position 2 (t=100h), nearly all mass should be on token 2.
    assert out[0, 2, 0].item() > 1.5
    assert out[0, 2, 1].abs().item() < 0.2


def test_causal_attention_all_padded_rows_stay_finite() -> None:
    from ehr_hier.transformer.encoder import AETCausalAttention

    attn = AETCausalAttention(
        d_model=8,
        num_heads=2,
        rope_module=_IdentityRoPE(),
        dropout=0.0,
        enable_alibi_hours_bias=False,
    )

    x = torch.randn(2, 6, 8)
    times = torch.zeros(2, 6, dtype=torch.float32)
    # Batch item 0 is fully padded; item 1 has real tokens.
    mask = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.long,
    )

    out = attn(x, times, mask)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[0], torch.zeros_like(out[0]))
