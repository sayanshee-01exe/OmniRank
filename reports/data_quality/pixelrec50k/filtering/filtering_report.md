# Filtering report - pixelrec50k

Enabled: **True** · converged: **True** · iterations: **1**

## Configuration

| Setting | Value |
|---|---:|
| `min_user_interactions` | 3 |
| `min_item_interactions` | 2 |
| `iterative` | True |

## Population before filtering

| Metric | Value |
|---|---:|
| Users | 50,000 |
| Items | 82,865 |
| Interactions | 989,494 |
| Singleton items | 13,518 |
| Items below item threshold | 13,518 |
| Users below user threshold | 0 |

## Iterations

| # | Users removed | Items removed | Interactions removed | Users left | Items left | Interactions left |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 13,518 | 13,518 | 50,000 | 69,347 | 975,976 |

## After filtering

Users **50,000** · items **69,347** · interactions **975,976**
