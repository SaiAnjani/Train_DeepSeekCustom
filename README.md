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

## Parameters
    Total Parameters: 25,373,440
    Trainable Parameters: 25,373,440
    
    Layer-wise parameters:
    --------------------------------------------------
    embed_tokens: 12,582,912 parameters
    layers: 12,790,272 parameters
      └─ 0: 2,131,712 parameters
      └─ 1: 2,131,712 parameters
      └─ 2: 2,131,712 parameters
      └─ 3: 2,131,712 parameters
      └─ 4: 2,131,712 parameters
      └─ 5: 2,131,712 parameters
    norm: 256 parameters
    lm_head: 12,582,912 parameters
    ==================================================
    Loaded 341094 tokens
    Vocabulary size: 49152



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

## Logs
    Step 500/10001 | Loss: 6.5418 | dt: 146.33ms | tok/sec:  874.72
    Step 1000/10001 | Loss: 5.1189 | dt: 143.01ms | tok/sec:  895.05
    Step 1500/10001 | Loss: 4.7368 | dt: 215.59ms | tok/sec:  593.73
    Step 2000/10001 | Loss: 6.4701 | dt: 144.47ms | tok/sec:  886.01
    Step 2500/10001 | Loss: 5.4151 | dt: 141.82ms | tok/sec:  902.53
    Step 3000/10001 | Loss: 5.8982 | dt: 137.99ms | tok/sec:  927.61
    Step 3500/10001 | Loss: 5.3119 | dt: 142.92ms | tok/sec:  895.61
    Step 4000/10001 | Loss: 5.1425 | dt: 177.16ms | tok/sec:  722.52
    Step 4500/10001 | Loss: 5.3831 | dt: 143.16ms | tok/sec:  894.09
    Step 5000/10001 | Loss: 5.1892 | dt: 134.66ms | tok/sec:  950.52
    Step 5500/10001 | Loss: 4.6916 | dt: 151.91ms | tok/sec:  842.58
    Step 6000/10001 | Loss: 4.6858 | dt: 144.54ms | tok/sec:  885.55
    Step 7500/10001 | Loss: 3.7104 | dt: 142.56ms | tok/sec:  897.85
    Step 8000/10001 | Loss: 4.7647 | dt: 217.34ms | tok/sec:  588.93
    Step 8500/10001 | Loss: 5.1420 | dt: 178.10ms | tok/sec:  718.70
    Step 9000/10001 | Loss: 4.0006 | dt: 138.61ms | tok/sec:  923.46
    Step 9500/10001 | Loss: 5.1212 | dt: 140.47ms | tok/sec:  911.24
    Step 10000/10001 | Loss: 4.9295 | dt: 134.74ms | tok/sec:  949.96

## Generated samples
    === Generated Samples ===

    Sample 1:
    --------------------------------------------------
    <|endoftext|> in the meets
    our with her taken my poor dismiss we fear,
    Good, no a Nature
    My lord will any as I have you
    As
     any!
    
    
    My bad,
    And it beern,
    Yourforts loss now-often: but stands thanks e upShe he Some for the matters way. aOL, course,
    So now you.
    
    As now you
    --
    
    You isle would she would she appeared?
    If you on were
    --------------------------------------------------
    
    Sample 2:
    --------------------------------------------------
    <|endoftext|>!
    
     oath I am ability.
    Will you would tender on leave
    I am a obedient, but that's tale
    To have sheina promisedhimAnd do you would make thy man
    You have lightione.
    Would not not qualify more, were you to be An then my see so drop much singular heir
    He from her in the
    
     war that
    For they with the saint,
    As,
    You the behalf not, in, which I am shores
    
    My
    --------------------------------------------------
    
    Sample 3:
    --------------------------------------------------
    <|endoftext|>
    Is anySome. O, and theurse;
    To give him, ared thee
     known a sens you equipped whose she steal.
    
    CAM
    
     uponGood manween.
    I am not not a man.
    She desire in removed you than the prince.
    You have has amore sunPA not
    And, what which
    Andio, yet my pl not
    Ay,
    What I would not his walk.
    C give him behalf thouLEer I have
    --------------------------------------------------
    
    Sample 4:
    --------------------------------------------------
    <|endoftext|>
    As you foundw new nor being home of stir
    As with me eye.
    
    LEONTES:
    The father is present heard? I'll makees
    Didst
    POLia hast more last, in she world,
    Well,
    Do, it now then, my ' none with word?
    Al of a forget of that she will I have that I may not a pieces; but you, to say you is to not fail:
    Th
    As alters I'll
    --------------------------------------------------
    
    Sample 5:
    --------------------------------------------------
    <|endoftext|>.
    
    PAULINA:
    No,No'er you: I had beresh.
    
    Clow you:
    My resemblance not over, d since
    And you reason ascore.
    As I make hermuch liking me sorry she swear
     threeSheMaster hear you.
    So in my lord,
    As you, I have affection
    ress! Pol you that poor we boundless,
    What,
     sorrow the draw; more talko, to give then you like your
    --------------------------------------------------
    =====================



    

