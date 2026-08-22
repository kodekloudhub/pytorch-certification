# PyTorch Certified Associate (PTCA)

Hands-on demos and labs for KodeKloud's PyTorch Certified Associate course.

The repository follows the Continued Learning course layout:

- `section_1`: PyTorch fundamentals
- `section_2`: data handling
- `section_3`: model development
- `section_4`: training optimization
- `section_5`: PTCA certification review

Content retained from the original PyTorch course stays in its original demo or
lab directory. PTCA-specific additions are placed alongside that material so a
learner can move through the course in lecture order.

Production planning for retained slides and recordings is tracked in
[`VIDEO_UPDATE_PLAN.md`](VIDEO_UPDATE_PLAN.md). It records the preservation,
insert, replacement, and asset-recovery action for every video and demo row in
the PTCA spreadsheet.

## Environment

The existing dependency versions are intentionally retained during this code
conversion. Install them with:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

