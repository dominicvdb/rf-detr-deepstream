"""Debug: how does DeepStream's RGB frame (nvinfer input, preprocessing in the ONNX) differ from OpenCV's?

Needs data/ds_dump2/ip_tensor_dump.bin (nvinfer dump-input-tensor=1 with models/nano_1280x720, net-scale-factor=1).
"""

import cv2
import numpy as np

H, W = 720, 1280
dump = np.memmap("data/ds_dump2/ip_tensor_dump.bin", dtype=np.float32, mode="r").reshape(-1, 3, H, W)
n = dump.shape[0]
cap = cv2.VideoCapture("data/sample_720p.mp4")
ocv = np.stack([cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) for _ in range(n)])
ds = np.asarray(dump)
print(f"frames: {n}   DeepStream value range [{ds.min():.1f}, {ds.max():.1f}]   OpenCV range [{ocv.min():.0f}, {ocv.max():.0f}]")

d = ds - ocv
print("\n1. raw difference (ds - opencv), 0-255 units")
print(f"   mean |diff| R,G,B = {np.abs(d).mean(axis=(0, 2, 3)).round(2)}   signed = {d.mean(axis=(0, 2, 3)).round(2)}")
print(f"   99th pct |diff| = {np.percentile(np.abs(d), 99):.1f}")

print("\n2. spatial shift? mean |diff| when OpenCV is shifted by (dy, dx) pixels")
for dy, dx in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)):
    a = ds[:, :, 2:-2, 2:-2]
    b = ocv[:, :, 2 + dy : H - 2 + dy, 2 + dx : W - 2 + dx]
    print(f"   dy={dy:+d} dx={dx:+d}: {np.abs(a - b).mean():.3f}")

print("\n3. colour conversion? least-squares fit  ds_rgb = A @ opencv_rgb + b  (per pixel)")
X = ocv.transpose(0, 2, 3, 1).reshape(-1, 3)[::50]
Y = ds.transpose(0, 2, 3, 1).reshape(-1, 3)[::50]
Xh = np.hstack([X, np.ones((len(X), 1), np.float32)])
coef, *_ = np.linalg.lstsq(Xh, Y, rcond=None)
A, b = coef[:3].T, coef[3]
resid = Y - Xh @ coef
print("   A =\n", A.round(4), "\n   b =", b.round(2))
print(f"   mean |diff| after fit = {np.abs(resid).mean():.3f}   (was {np.abs(Y - X).mean():.3f})")

print("\n4. luma (brightness) vs chroma (colour): which one differs? BT.601 weights")


def luma_chroma(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y, b - y, r - y  # Y, ~Cb, ~Cr (unscaled)


y_ds, cb_ds, cr_ds = luma_chroma(ds)
y_cv, cb_cv, cr_cv = luma_chroma(ocv)
print(f"   mean |dY| = {np.abs(y_ds - y_cv).mean():.3f}   mean |dCb| = {np.abs(cb_ds - cb_cv).mean():.3f}"
      f"   mean |dCr| = {np.abs(cr_ds - cr_cv).mean():.3f}")
for dy, dx in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)):
    sl_a = (slice(None), slice(2, -2), slice(2, -2))
    sl_b = (slice(None), slice(2 + dy, H - 2 + dy), slice(2 + dx, W - 2 + dx))
    print(f"   shift dy={dy:+d} dx={dx:+d}:  |dY|={np.abs(y_ds[sl_a] - y_cv[sl_b]).mean():.3f}"
          f"  |dCb|={np.abs(cb_ds[sl_a] - cb_cv[sl_b]).mean():.3f}  |dCr|={np.abs(cr_ds[sl_a] - cr_cv[sl_b]).mean():.3f}")

print("\n5. does DeepStream resample at scale 1? compare with OpenCV averaged over neighbours (half-pixel offsets)")
y = y_cv
inner = (slice(None), slice(2, -2), slice(2, -2))
cands = {
    "identity": y,
    "avg x, x-1": 0.5 * (y + np.roll(y, 1, axis=2)),
    "avg x, x+1": 0.5 * (y + np.roll(y, -1, axis=2)),
    "avg y, y-1": 0.5 * (y + np.roll(y, 1, axis=1)),
    "avg 2x2 (x-1,y-1)": 0.25 * (y + np.roll(y, 1, 2) + np.roll(y, 1, 1) + np.roll(np.roll(y, 1, 1), 1, 2)),
    "avg 2x2 (x+1,y+1)": 0.25 * (y + np.roll(y, -1, 2) + np.roll(y, -1, 1) + np.roll(np.roll(y, -1, 1), -1, 2)),
}
for name, c in cands.items():
    print(f"   {name:20s} |dY| = {np.abs(y_ds[inner] - c[inner]).mean():.3f}")
