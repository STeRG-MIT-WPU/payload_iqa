"""Host resource sampling (CPU / RAM) via psutil.

TPU load has no hardware counter on the Coral; the workers report a
duty-cycle instead (see workers._DutyMeter) — this module only covers the
host side.
"""
try:
    import psutil
    psutil.cpu_percent(None)   # prime the counter
except ImportError:
    psutil = None


class ResourceMonitor:
    available = psutil is not None

    def sample(self):
        """Return {'cpu': %, 'per_core': [...], 'ram': %, 'ram_used_gb': x,
        'ram_total_gb': y} or None if psutil is missing."""
        if psutil is None:
            return None
        vm = psutil.virtual_memory()
        return {
            "cpu": psutil.cpu_percent(None),
            "per_core": psutil.cpu_percent(None, percpu=True),
            "ram": vm.percent,
            "ram_used_gb": vm.used / 2**30,
            "ram_total_gb": vm.total / 2**30,
        }
