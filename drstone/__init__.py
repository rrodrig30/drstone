"""Dr Stone — kidney-stone composition prediction from non-contrast CT.

A module on the PRISM core: reuses DICOM loading, interpolation, segmentation,
radiomics, 3D reconstruction, and the database/web layers, and adds stone-specific
segmentation, Hb-anchored HU calibration, clinical/lab intake, FTIR/XRD label
handling, and the multimodal mixed-effects composition model.

Phase 0 (this commit): data engineering — join the project spreadsheet
(Sheet1 clinical/labs ↔ COMP composition labels ↔ MRN crossover) on UT MRN and
emit a clean per-stone analytic table.
"""

__all__ = ["config"]
