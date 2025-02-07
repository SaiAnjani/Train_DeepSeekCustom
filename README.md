# DeepSeek Architecture Implementation

This repository implements a DeepSeek-style architecture featuring Multi-Query Attention with Local Head Attention (MLHA) and Mixture of Experts (MoE) with loss-less load balancing.

## Architecture Overview

The architecture combines two key innovations:
1. MLHA (Multi-Query with Local Head Attention)
2. MoE (Mixture of Experts) with loss-less load balancing

### Model Configuration
    @dataclass
    class DeepSeekConfig:
    block_size: int = 128
    vocab_size: int = vocab_size
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    intermediate_size: int = 512
    # MLHA settings
    num_key_value_heads: int = 2 # Compression ratio of 8:1
    window_size: int = 32 # Local attention window
    # MoE settings
    num_experts: int = 4
    num_shared_experts: int = 1
    top_k_experts: int = 2
    expert_capacity_factor: float = 1.25




## 1. Multi-Query with Local Head Attention (MLHA)

MLHA combines three key concepts:
- Multi-Query Attention
- Local Head Attention
- Sliding Window Attention

### Key Features

1. **Multi-Query Attention**:
   - Uses fewer key-value heads than query heads
   - Reduces memory usage while maintaining performance
   - Implementation:
   ```python
   # Compressed key-value heads
   num_key_value_heads = num_heads // 8  # 8:1 compression
   k = k.repeat_interleave(num_key_value_groups, dim=2)
   v = v.repeat_interleave(num_key_value_groups, dim=2)
   ```

2. **Local Head Attention**:
   - Processes local context with sliding windows
   - Improves efficiency for nearby token relationships
   - Implementation:
   ```python
   # Local attention with sliding window
   for i in range(0, seq_len, window_size):
       end_idx = min(i + window_size, seq_len)
       local_q = q[:, :, i:end_idx, :]
       local_k = k[:, :, max(0, i-window_size//2):min(seq_len, end_idx+window_size//2), :]
       local_v = v[:, :, max(0, i-window_size//2):min(seq_len, end_idx+window_size//2), :]
   ```

3. **Combined Attention**:
   - Merges global and local attention
   - Balances long-range and local dependencies
   ```python
   # Combine global and local attention
   out = 0.5 * (global_out + local_out)
   ```

## 2. Mixture of Experts (MoE) with Loss-less Load Balancing

MoE increases model capacity without proportional compute cost increase.

### Components

1. **Expert Networks**:
   - Multiple specialized neural networks
   - Shared experts for common patterns
   - Implementation:
   ```python
   self.experts = nn.ModuleList([
       LlamaMLP(config) for _ in range(total_experts)
   ])
   ```

2. **Dynamic Routing**:
   - Routes tokens to most relevant experts
   - Uses top-k routing for redundancy
   ```python
   # Select top-k experts for each token
   top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
   ```

3. **Loss-less Load Balancing**:
   - Ensures even distribution of tokens across experts
   - Prevents expert collapse
   - Implementation:
   ```python
   # Calculate load balancing loss
   target_count = (batch_size * seq_len * top_k) / num_experts
   load_balancing_loss = torch.sum((expert_counts - target_count)**2) / 
                        (num_experts * target_count**2)
   ```

### Load Balancing Features

1. **Capacity Control**:
   - Uses capacity factor to prevent overload
   - Implements soft routing with probability weighting
   ```python
   capacity = int(capacity_factor * tokens_per_expert)
   ```

2. **Expert Utilization**:
   - Tracks expert usage
   - Balances load through loss function
   ```python
   expert_counts = torch.zeros(num_experts, device=x.device)
   for expert_idx in range(num_experts):
       expert_mask = (expert_indices == expert_idx)
       expert_counts[expert_idx] = expert_mask.sum()
   ```

## Benefits

1. **Efficiency**:
   - Reduced memory usage through key-value head compression
   - Efficient local context processing
   - Dynamic computation routing

2. **Performance**:
   - Better handling of both local and global dependencies
   - Increased model capacity through expert specialization
   - Balanced computational load

3. **Scalability**:
   - Architecture scales well with model size
   - Efficient resource utilization
   - Prevents bottlenecks through load balancing

## Usage
    python
    Initialize model
    model = DeepSeekLM(DeepSeekConfig())
    Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    Forward pass includes both CE loss and load balancing loss
    logits, loss = model(x, y)
    Generate text
    generated_text = model.generate(
    context,
    max_new_tokens=100,
    temperature=0.8
    )
