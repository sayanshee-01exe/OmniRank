# Missing-modality evaluation

How the item tower behaves when an item has text but no image, an image but no
text, or neither — and what the real corpus lets us claim about it.

## The mechanism

Missing modalities are represented by a **learned token per modality**, not by
a zero vector:

```python
class ModalityEncoder(nn.Module):
    def forward(self, features, present):
        encoded = self.project(features)
        return torch.where(present.unsqueeze(-1), encoded, self.missing)
```

A zero vector would be a specific point in the projected space — one the model
did not choose and cannot move — and it collides with whatever legitimately
projects near the origin. The learned `missing` parameter is trained like any
other, so "I have no image" becomes a representation the model picked.

Consequences worth stating:

- An item with no modalities at all still produces a well-defined embedding
  (the sum of the learned missing tokens, plus tag and residual terms). It is
  not a special case in the scoring path.
- Because the token is shared across all items missing that modality, those
  items are pulled toward each other. That is a real effect, not a bug: with no
  content signal there is nothing to separate them by.

## What the real corpus can and cannot show

PixelRec50K, after k-core filtering, has **complete coverage of both
modalities**:

| Group | Items |
| --- | ---: |
| both modalities | 69,347 |
| text only | 0 |
| image only | 0 |
| neither | 0 |

So the missing-modality views over real data are **empty**, and
`reports/metrics/phase_05/missing_modality_metrics.csv` reflects that.

**This path is therefore not exercised on real data.** It is not reported as
"robust to missing modalities", because on this corpus that claim has no
measurement behind it. The alternative — dropping modalities artificially to
manufacture a number — would report a property of the ablation, labelled as a
property of the data.

## How it is verified instead

Through fixtures, in `tests/unit/models/two_tower/test_towers.py`:

- an item with no text and no image still yields a finite embedding;
- the missing token is used exactly where `present` is false, and the projected
  feature exactly where it is true;
- the missing token receives gradient (it is learned, not frozen at
  initialisation);
- two items differing only in which modality is absent do not collapse to the
  same vector.

These establish that the code does what it claims. They do not establish how
much retrieval quality survives on a corpus with real gaps — that needs a
corpus with real gaps.

## When this becomes a live measurement

A dataset with genuine partial coverage, or PixelRec with the k-core filter
relaxed enough to admit items whose media is absent. At that point the empty
views populate and the numbers mean something. Until then this document
records a capability with a stated, unmeasured status.

See [cold_item_evaluation.md](cold_item_evaluation.md) for the related but
distinct case: items with full content and *no interactions*.
