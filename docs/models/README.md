# Model documentation

| Document | Covers |
|---|---|
| [`popularity.md`](popularity.md) | Global and time-decayed popularity; why it is required |
| [`bpr_matrix_factorization.md`](bpr_matrix_factorization.md) | BPR objective, retrieval, device policy, persistence |
| [`negative_sampling.md`](negative_sampling.md) | Sampling guarantees and the vectorised implementation |
| [`model_selection.md`](model_selection.md) | Validation/test discipline and the configuration lock |
| [`lightgcn.md`](lightgcn.md) | Graph propagation, the `num_layers=0` ablation, isolated nodes |
| [`sasrec.md`](sasrec.md) | Causal attention, padding, sampled BCE, and what the budget allowed |

## Implementation status

| Model | Phase | Status |
|---|---|---|
| Popularity (global, time-decay) | 3 | ✅ Implemented |
| BPR matrix factorization | 3 | ✅ Implemented |
| LightGCN | 4 | ✅ Implemented |
| SASRec | 4 | ✅ Implemented |
| Blended retriever (fusion) | 4 | ✅ Implemented |
| Two-tower multimodal | 5 | Not implemented |
| LightGBM ranker | 6 | Not implemented |
| MMR reranker | 6 | Not implemented |
