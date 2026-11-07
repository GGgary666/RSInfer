# Probe 7 (CPU only): reload a module by rewriting its file in place and re-importing.
# CPython validates a cached .pyc by the source's mtime (whole seconds) and size, so a
# rewrite within the same second that keeps the size imports the OLD code.
# usage: python p7_stale_bytecode.py
import importlib, json, sys, tempfile, time

d = tempfile.mkdtemp(prefix="p7_")
sys.path.insert(0, d)


def write(v):
    with open(f"{d}/p7_mod.py", "w") as f:
        f.write(f"VERSION = {v}\n")


def reimport():
    sys.modules.pop("p7_mod", None)
    importlib.invalidate_caches()
    return importlib.import_module("p7_mod").VERSION


write(1); first = reimport()
write(2); same_second = reimport()
time.sleep(1.1)
write(3); later = reimport()
sys.dont_write_bytecode = True
d2 = tempfile.mkdtemp(prefix="p7_nobc_")
sys.path.insert(0, d2)
with open(f"{d2}/p7_nobc.py", "w") as f:
    f.write("VERSION = 1\n")
sys.modules.pop("p7_nobc", None); importlib.invalidate_caches(); a = importlib.import_module("p7_nobc").VERSION
with open(f"{d2}/p7_nobc.py", "w") as f:
    f.write("VERSION = 2\n")
sys.modules.pop("p7_nobc", None); importlib.invalidate_caches(); b = importlib.import_module("p7_nobc").VERSION
print("RESULT " + json.dumps(dict(
    wrote=[1, 2, 3], imported=[first, same_second, later],
    same_second_rewrite_imports_stale_code=same_second != 2,
    without_bytecode_cache=[a, b], python=sys.version.split()[0])))
