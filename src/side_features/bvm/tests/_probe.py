import side_features.bvm.connect as C
_orig = C.connect
_seen = []
def spy(a, b, eps_stage=1e-2, efficiency=1.0):
    r = _orig(a, b, eps_stage, efficiency)
    if r["connected"]:
        _seen.append((r["dmin"], r["tol"]))
    return r
C.connect = spy
import atexit, numpy as np
@atexit.register
def dump():
    if not _seen: return
    d = np.array(_seen)
    print("\nCONNECTED dmin: max %.4f  p95 %.4f  median %.4f | tol max %.4f" %
          (d[:,0].max(), np.percentile(d[:,0],95), np.median(d[:,0]), d[:,1].max()))
    big = d[d[:,0] > 0.05]
    print("connected with dmin>0.05:", len(big), "of", len(d))
    for row in sorted(big.tolist(), reverse=True)[:15]: print("  dmin %.4f tol %.4f" % tuple(row))
