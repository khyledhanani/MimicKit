# GMR to MimicKit Motion Data Conversion Guide

This guide demonstrates how to convert motion data from GMR format to MimicKit format and verify the contents of both input and output files.

**Prerequisites:** 

- All commands should be run from the root directory of the repository (`MimicKit/`).


---

## Convert GMR to MimicKit Format

Use the conversion script to transform your GMR motion data into MimicKit-compatible format.

**Command:**

```bash
python tools/gmr_to_mimickit/gmr_to_mimickit.py --input_file {input_file_path} --output_file {output_file_path}
```

---

## Note:

The output motion file can be placed anywhere, but if you want to wire it into the checked-in configs it is convenient to keep it under `data/motions/`. After conversion, validate it against the target rig with `tools/validate_mimickit_motion.py` before using it in an env config.
