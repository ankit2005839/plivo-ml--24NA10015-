# Run Log

| Run ID | Architecture Details | Vocab Size | Context Window | Optim Strategy | Final Loss | Dev Eval BPB |
|--------|----------------------|------------|----------------|----------------|------------|--------------|
| Baseline | 4 Layers, 160 Hidden | 256 (Byte) | 128 | Adam (Fixed LR)| ~4.1200 | ~3.4500 |
| Run 01   | 3 Layers, 192 Hidden | 2048 (BPE) | 256 | AdamW + Cosine | ~1.9400 | **[Your Score]** |