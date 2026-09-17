from pathlib import Path

p = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\output\otr\obs"
    r"\the_christening_gown_beneath_the_door_20260915_162604_silent"
    r"__vart__cvdu__clum__elev__orig__cmsa__soni_final.mp4"
)
print(f"exists={p.exists()} bytes={p.stat().st_size if p.exists() else 0} path={p}")
