# Metrics

SID4=2122, SEED=2122, SLICE=122, HP_ID=4, CLS_A=2, CLS_B=8

| Method | Epochs | Final train acc | Final test acc |
|---|---|---|---|
| A: Supervised (500 labels) | 15 | 0.7120 | 0.3639 |
| B: Rotation SSL + linear (500 labels) | 15 pretrain + 20 linear | 0.2180 | 0.1921 |
| B: Rotation SSL + linear, lr 1e-2 (500 labels) | 15 pretrain + 20 linear | 0.3600 | 0.3762 |
| C: SimCLR + linear, lr 1e-2 (500 labeled images) | 20 pretrain + 20 linear | 0.7040 | 0.5320 |

| Encoder | Top 5 neighbor precision (all 8000 test images) |
|---|---|
| Supervised | 0.3670 |
| Rotation SSL | 0.3352 |
| SimCLR | 0.4819 |
