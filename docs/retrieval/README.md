# Retrieval documentation

| Document | Covers |
|---|---|
| [`candidate_aggregation.md`](candidate_aggregation.md) | The three fusion strategies, over-retrieval, and the audit trail |
| [`reciprocal_rank_fusion.md`](reciprocal_rank_fusion.md) | The RRF arithmetic, choosing `c`, and why rank beats score |
| [`faiss_index.md`](faiss_index.md) | Exactness checking, identity enforcement, bounded exclusion search |
| [`two_tower_faiss.md`](two_tower_faiss.md) | Indexing two-tower embeddings, tie-aware brute-force verification |
| [`five_source_fusion.md`](five_source_fusion.md) | Adding the two-tower to the blend, and what it actually contributes |

## Implementation status

| Component | Phase | Status |
|---|---|---|
| Weighted round robin | 4 | ✅ Implemented |
| Reciprocal rank fusion | 4 | ✅ Implemented |
| Normalised score union | 4 | ✅ Implemented |
| Blended retriever | 4 | ✅ Implemented |
| FAISS flat index (exact) | 4 | ✅ Implemented |
| FAISS HNSW / IVF (approximate) | 4 | ✅ Implemented, benchmarked against exact |
| pgvector / managed index | later | Not implemented (ADR-004) |

## The two questions this stage has to answer

Accuracy metrics measure the ranked list. They do not answer either of the
questions that decide whether a multi-source retrieval stage earns its
complexity:

1. **Can the candidates contain the answer at all?** See
   [candidate recall](../evaluation/candidate_recall.md). This is the ceiling
   every downstream stage inherits, and no ranker can raise it.
2. **Are the sources actually different?** See the same document's section on
   source overlap. Four generators returning the same list cost four times the
   compute for one generator's coverage.
