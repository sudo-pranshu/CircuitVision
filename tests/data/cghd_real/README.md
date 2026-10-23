# Real CGHD annotation fixture

`C145_D1_P1.xml` is one real annotation file from CGHD (drafter 13), as published in the
dataset's Hugging Face copy (`lowercaseonly/cghd`, file
`drafter_13/annotations/C145_D1_P1.xml`), retrieved 2026-10-23. It is not the MD5-verified
Zenodo v14 archive.

It is kept as a regression fixture because it shows the real annotation format. In
particular, `<rotation>` is nested inside `<bndbox>` and has 1-degree resolution (269).

CGHD: J. Bayer, F. Thoma et al., *A Public Ground-Truth Dataset for Handwritten Circuit
Diagram Images*, licensed CC BY 3.0.
