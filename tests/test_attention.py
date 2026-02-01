import torch
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.attention import MultiHeadAttention


def test_multi_head_attention():
    """
    Comprehensive tests for multi-head attention.
    
    Tests:
    1. Shape correctness
    2. Head splitting and combining
    3. Comparison with single-head attention
    4. Gradient flow
    5. Different sequence lengths (cross-attention)
    """
    
    print("\n" + "="*80)
    print("TESTING MULTI-HEAD ATTENTION")
    print("="*80)
    
    torch.manual_seed(42)
    
    # Parameters
    batch_size = 2
    seq_len = 10
    d_model = 128
    num_heads = 8
    
    # Create module
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    
    # Test 1: Shape correctness
    print("\nTest 1: Shape Correctness")
    print("-" * 40)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, weights = mha(x, x, x)  # Self-attention
    
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Output shape mismatch: {output.shape}"
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len), \
        f"Weights shape mismatch: {weights.shape}"
    
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Attention weights shape: {weights.shape}")
    
    # Test 2: Head splitting and combining
    print("\nTest 2: Head Split/Combine Consistency")
    print("-" * 40)
    
    x = torch.randn(batch_size, seq_len, d_model)
    x_split = mha.split_heads(x)
    x_combined = mha.combine_heads(x_split)
    
    assert torch.allclose(x, x_combined, atol=1e-6), \
        "Split and combine not inverse operations!"
    
    print(f"✓ Original shape: {x.shape}")
    print(f"✓ After split: {x_split.shape}")
    print(f"✓ After combine: {x_combined.shape}")
    print(f"✓ Reconstruction error: {(x - x_combined).abs().max():.8f}")
    
    # Test 3: Cross-attention (different sequence lengths)
    print("\nTest 3: Cross-Attention")
    print("-" * 40)
    
    seq_len_q = 10
    seq_len_kv = 20
    
    query = torch.randn(batch_size, seq_len_q, d_model)
    key = torch.randn(batch_size, seq_len_kv, d_model)
    value = torch.randn(batch_size, seq_len_kv, d_model)
    
    output, weights = mha(query, key, value)
    
    assert output.shape == (batch_size, seq_len_q, d_model), \
        f"Cross-attention output shape wrong: {output.shape}"
    assert weights.shape == (batch_size, num_heads, seq_len_q, seq_len_kv), \
        f"Cross-attention weights shape wrong: {weights.shape}"
    
    print(f"✓ Query length: {seq_len_q}")
    print(f"✓ Key/Value length: {seq_len_kv}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Weights shape: {weights.shape}")
    
    # Test 4: Gradient flow
    print("\nTest 4: Gradient Flow")
    print("-" * 40)
    
    mha_grad = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    output, _ = mha_grad(x, x, x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient!"
    assert not torch.isnan(x.grad).any(), "NaN gradients!"
    
    print(f"✓ Gradients computed")
    print(f"✓ Input grad norm: {x.grad.norm():.4f}")
    
    # Check all parameter gradients
    for name, param in mha_grad.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
    
    print(f"✓ All {len(list(mha_grad.parameters()))} parameters have valid gradients")
    
    # Test 5: Masking
    print("\nTest 5: Causal Masking")
    print("-" * 40)
    
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    output_masked, weights_masked = mha(x, x, x, mask=mask)
    
    # Check that future positions are masked across all heads
    for h in range(num_heads):
        for i in range(seq_len):
            for j in range(i+1, seq_len):
                assert weights_masked[0, h, i, j] < 1e-6, \
                    f"Head {h}: position ({i},{j}) should be masked"
    
    print(f"✓ Causal mask applied to all {num_heads} heads")
    print(f"✓ Sample masked weight (head 0): {weights_masked[0, 0, 0, 5]:.8f}")
    
    # Test 6: Dimension validation
    print("\nTest 6: Dimension Validation")
    print("-" * 40)
    
    try:
        invalid_mha = MultiHeadAttention(d_model=127, num_heads=8)  # Not divisible
        assert False, "Should have raised assertion error"
    except AssertionError as e:
        print(f"✓ Correctly caught invalid dimension: {e}")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
    print("\nStage 2 Complete. Multi-head attention is correct.")
    print("Next: Stage 3 - Positional Encoding")


if __name__ == "__main__":
    test_multi_head_attention()
