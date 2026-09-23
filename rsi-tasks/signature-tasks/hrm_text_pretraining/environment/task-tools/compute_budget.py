"""Pure-Python compute-clock, conservative stopping and calibration arithmetic."""
import math
import threading
import time

CLOCK = "synchronized_training_minus_compile"


def interval_union_seconds(intervals, start, end):
    """Clip and merge intervals so nested/overlapping compilations count once."""
    if end < start:
        raise ValueError("Clock moved backwards")
    clipped = sorted((max(start, a), min(end, b)) for a, b in intervals if b > start and a < end)
    total, previous = 0.0, start
    for left, right in clipped:
        if right < left:
            raise ValueError("Invalid compilation interval")
        total += max(0.0, right - max(previous, left))
        previous = max(previous, right)
    return total


class CompilationTracker:
    """Use TorchDynamo callbacks, including lazy backward/cudagraph compilation.

    The runner synchronizes CUDA before begin() and before finish(). The charged
    interval includes numerical dispatch and distributed communication, excludes
    data transfer, and subtracts compiler callback intervals. This is a training
    critical-path clock, not the sum of CUDA kernel durations.
    """
    def __init__(self, now=time.perf_counter):
        self.now = now
        self.lock = threading.Lock()
        self.depth = 0
        self.intervals = []
        self.start = None
        self.compile_start = None

    def begin(self):
        with self.lock:
            if self.depth or self.start is not None:
                raise RuntimeError("Compilation/update interval is already active")
            self.intervals = []
            self.start = self.now()

    def on_compile_start(self, *_args):
        with self.lock:
            if self.depth == 0:
                self.compile_start = self.now()
            self.depth += 1

    def on_compile_end(self, *_args):
        with self.lock:
            if self.depth <= 0:
                raise RuntimeError("Unmatched compilation callback")
            self.depth -= 1
            if self.depth == 0:
                self.intervals.append((self.compile_start, self.now()))
                self.compile_start = None

    def finish(self):
        with self.lock:
            if self.start is None or self.depth:
                raise RuntimeError("Incomplete update/compilation interval")
            end = self.now()
            elapsed = end - self.start
            compilation = interval_union_seconds(self.intervals, self.start, end)
            self.start = None
            return {"training_region_seconds": elapsed, "compilation_seconds": compilation,
                    "compute_seconds": max(0.0, elapsed - compilation),
                    "compilation_events": len(self.intervals)}


class ComputeBudget:
    """Reserve a conservative predicted update before starting it.

    No finite timing sample can bound arbitrary future candidate execution. An
    unexpectedly overlong update is reported as a budget violation, never hidden
    or relabeled as a valid capped checkpoint.
    """
    def __init__(self, limit, used=0.0, initial_step_bound=1.0, phase_max=None):
        if not all(math.isfinite(value) and value >= 0 for value in (limit, used, initial_step_bound)) or limit <= 0:
            raise ValueError("Invalid compute budget")
        self.limit, self.used = float(limit), float(used)
        self.initial_step_bound = max(float(initial_step_bound), 0.05)
        self.phase_max = dict(phase_max or {})
        if any(not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for key, value in self.phase_max.items()):
            raise ValueError("Invalid resumed phase timing history")

    def step_reserve(self, phase=None):
        key = str(phase)
        observed = self.phase_max.get(key)
        if observed is None:
            observed = max(self.phase_max.values(), default=self.initial_step_bound)
            # An unseen backward depth can retain more activations/work.
            observed = max(self.initial_step_bound, observed * 2)
        return observed * 1.5 + 0.05

    def can_start(self, phase=None):
        return self.limit - self.used >= self.step_reserve(phase)

    def reserve_evidence(self, phase=None):
        """Expose every input needed to reproduce the pre-update stop decision."""
        return {"method": "phase_observed_max_with_unseen_phase_fallback", "phase": str(phase),
                "phase_max_seconds": dict(self.phase_max), "initial_step_bound_seconds": self.initial_step_bound,
                "unseen_phase_multiplier": 2.0, "safety_multiplier": 1.5, "fixed_margin_seconds": 0.05,
                "seconds": self.step_reserve(phase)}

    def charge(self, seconds, phase=None):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid measured compute interval")
        self.used += seconds
        key = str(phase)
        self.phase_max[key] = max(self.phase_max.get(key, 0), seconds)
        return self.used <= self.limit


def retained_checkpoint_stop(budget, phase, *, has_checkpoint, completed_epochs, requested_epochs):
    """A valid retained checkpoint can be exported without another update."""
    if budget.used > budget.limit:
        raise ValueError("Retained experiment already exceeds the compute budget")
    if not has_checkpoint:
        return None
    if completed_epochs >= requested_epochs:
        return "epochs_completed"
    if not budget.can_start(phase):
        return "compute_budget"
    return None


def phase_step_counts(total_steps, minimum=2, maximum=6, warmup_ratio=0.5):
    """Exactly count the native compute_train_extra_args schedule on CPU."""
    if type(total_steps) is not int or total_steps < 1 or maximum < minimum or not 0 <= warmup_ratio <= 1:
        raise ValueError("Invalid backward schedule")
    result = {}
    warmup = total_steps * warmup_ratio
    for step in range(1, total_steps + 1):
        progress = min(1.0, step / warmup) if warmup > 0 else 1.0
        phase = minimum + int(progress * (maximum - minimum))
        result[phase] = result.get(phase, 0) + 1
    return result


def project_baseline(records, phase_counts, discard_per_phase=2):
    """Project measured representative phases; retain sample spread as uncertainty."""
    groups = {}
    for record in records:
        groups.setdefault(int(record.get("bp_steps", 0)), []).append(float(record["compute_seconds"]))
    phases, estimate, upper = {}, 0.0, 0.0
    for phase, count in phase_counts.items():
        samples = groups.get(int(phase), [])[discard_per_phase:]
        if len(samples) < 2 or any(not math.isfinite(value) or value <= 0 for value in samples):
            raise ValueError(f"Insufficient positive steady timing samples for phase {phase}")
        mean = sum(samples) / len(samples)
        phases[str(phase)] = {"scheduled_steps": count, "measured_steps": len(samples),
                              "mean_compute_seconds": mean, "min_compute_seconds": min(samples),
                              "max_compute_seconds": max(samples)}
        estimate += count * mean
        upper += count * max(samples)
    return {"estimated_baseline_compute_seconds": estimate,
            "sample_max_projection_seconds": upper, "phase_measurements": phases,
            "uncertainty_note": "Short phase samples; projection assumes stationary execution and an estimated packed update count. Sample maxima are not a statistical confidence bound."}
