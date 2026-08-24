"""PixelRec50K source adapter.

Everything that knows PixelRec's column names, id conventions, and quirks lives
in this package. The rest of the pipeline works on canonical frames and has no
idea which dataset produced them - which is what makes onboarding a second
vertical a new package here rather than edits everywhere.
"""

from __future__ import annotations

from omnirank.data.pixelrec.loaders import PixelRec50KLoader, RawPixelRec, SourceFile

__all__ = ["PixelRec50KLoader", "RawPixelRec", "SourceFile"]
