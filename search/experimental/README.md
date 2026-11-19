# Experimental (not used by the pipeline)

`program.py`: guarded-stage programs (conditional, closed-loop actions: "change lane, then if X go back, otherwise accelerate").
Shelved by decision: the pipeline keeps the original hand-defined families (`search/families.py` and the per-case FAM dicts). Nothing imports this directory.
`test_program.py` is a toy check that iCEM can find a conditional program without it being hard-coded.
