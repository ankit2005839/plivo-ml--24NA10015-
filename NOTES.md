 Engineering Notes

 1. Tokenizer Optimization
Swapped standard raw byte evaluation for a custom 2048-token Byte-Level BPE tokenizer. This eliminates context-window fragmentation on Hindi/Devanagari text sequences, increasing effective processing throughput.

2. Parameter Efficiencies
Used weight tying (`tie_weights = True`) between inputs and output projection layers to save parameter overhead. This allowed increasing the hidden dimension width to 192 while remaining under the 2M parameter cap.

3. Training Dynamics
Replaced fixed Adam configurations with AdamW decoupled decay optimizations, configured with a 150-step linear warmup followed by a Cosine decay curve.
