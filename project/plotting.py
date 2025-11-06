from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import hashlib
import pandas as pd
import matplotlib.pyplot as plt

try:
    from rich.console import Console
    from rich.table import Table
    _HAS_RICH = True
except Exception:
    _HAS_RICH = False


def _short_hash(text: Optional[str]) -> str:
    if not text:
        return ""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


@dataclass
class IterationLog:
    iteration: int
    change_type: str  # e.g., "temperature", "top_p", "top_k", "system_prompt", "combo", "none"
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    system_prompt_preview: Optional[str]        # small preview or descriptor
    system_prompt_hash: Optional[str]
    changed_fields: List[str]                   # e.g., ["temperature", "top_p"]
    test_accuracy: float                        # e.g., 0.72
    time_rag_s: float
    time_test_s: float
    time_interpret_s: float
    time_mechanic_s: float

    @property
    def time_total_s(self) -> float:
        return self.time_rag_s + self.time_test_s + self.time_interpret_s + self.time_mechanic_s


class RagTuningRun:
    """Collects per-iteration logs and renders final summary (tables + charts)."""

    def __init__(self, run_name: str = "rag_tuning_run"):
        self.run_name = run_name
        self._rows: List[IterationLog] = []

    # ----- add entries each iteration -----
    def log_iteration(
        self,
        iteration: int,
        change_type: str,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        system_prompt: Optional[str],
        changed_fields: List[str],
        test_accuracy: float,
        times: Dict[str, float],  # {"rag":..., "test":..., "interpret":..., "action":...}
    ):
        preview = (system_prompt or "")[:60].replace("\n", " ") if system_prompt else None
        entry = IterationLog(
            iteration=iteration,
            change_type=change_type,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            system_prompt_preview=preview,
            system_prompt_hash=_short_hash(system_prompt),
            changed_fields=changed_fields,
            test_accuracy=float(test_accuracy),
            time_rag_s=float(times.get("rag", 0.0)),
            time_test_s=float(times.get("test", 0.0)),
            time_interpret_s=float(times.get("interpret", 0.0)),
            time_mechanic_s=float(times.get("mechanic", 0.0)),
        )
        self._rows.append(entry)

    # ----- produce pandas dataframe -----
    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame([asdict(r) for r in self._rows])
        if df.empty:
            return df

        # order & derived cols
        cols = [
            "iteration", "change_type", "changed_fields",
            "temperature", "top_p", "top_k",
            "system_prompt_hash", "system_prompt_preview",
            "test_accuracy",
            "time_rag_s", "time_test_s", "time_interpret_s", "time_mechanic_s"
        ]
        df = df[cols]
        df["time_total_s"] = df[["time_rag_s", "time_test_s", "time_interpret_s", "time_mechanic_s"]].sum(axis=1)
        return df.sort_values("iteration").reset_index(drop=True)

    # ----- console pretty print -----
    def print_console_summary(self):
        df = self.to_dataframe()
        if df.empty:
            print("No iterations logged.")
            return

        # Best, totals
        best_idx = df["test_accuracy"].idxmax()
        best_row = df.loc[best_idx]
        total_time = df["time_total_s"].sum()

        print(f"\n=== {self.run_name} — Summary ===")
        print(f"Iterations: {len(df)} | Best accuracy: {best_row['test_accuracy']:.3f} (iter {int(best_row['iteration'])})")
        print(f"Total runtime: {total_time:.2f}s\n")

        # Rich table if available, else plain pandas print
        if _HAS_RICH:
            console = Console()
            table = Table(show_lines=False, title="Iterations")
            for col in ["iter", "change", "changed_fields", "temp", "top_p", "top_k", "prompt#",
                        "acc", "rag(s)", "test(s)", "interp(s)", "action(s)", "total(s)"]:
                table.add_column(col)

            for _, row in df.iterrows():
                table.add_row(
                    str(int(row["iteration"])),
                    str(row["change_type"]),
                    ", ".join(row["changed_fields"]) if row["changed_fields"] else "",
                    "" if pd.isna(row["temperature"]) else f"{row['temperature']:.2f}",
                    "" if pd.isna(row["top_p"]) else f"{row['top_p']:.2f}",
                    "" if pd.isna(row["top_k"]) else str(int(row["top_k"])),
                    row["system_prompt_hash"] or "",
                    f"{row['test_accuracy']:.3f}",
                    f"{row['time_rag_s']:.2f}",
                    f"{row['time_test_s']:.2f}",
                    f"{row['time_interpret_s']:.2f}",
                    f"{row['time_mechanic_s']:.2f}",
                    f"{row['time_total_s']:.2f}",
                )
            console.print(table)
        else:
            print(df.to_string(index=False))

    # ----- charts (accuracy line + stacked time bars) -----
    def plot_figures(self, show: bool = True, save_prefix: Optional[str] = None):
        df = self.to_dataframe()
        if df.empty:
            print("No data to plot.")
            return

        # Accuracy over iterations
        plt.figure()
        plt.plot(df["iteration"], df["test_accuracy"], marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Test Accuracy")
        plt.title(f"{self.run_name}: Accuracy per Iteration")
        if save_prefix:
            plt.savefig(f"{save_prefix}_accuracy.png", bbox_inches="tight")
        if show:
            plt.show()

        # Stacked bars for timings
        plt.figure()
        bottom = None
        for col in ["time_rag_s", "time_test_s", "time_interpret_s", "time_mechanic_s"]:
            if bottom is None:
                bars = plt.bar(df["iteration"], df[col], label=col)
                bottom = df[col].values
            else:
                bars = plt.bar(df["iteration"], df[col], bottom=bottom, label=col)
                bottom = bottom + df[col].values
        plt.xlabel("Iteration")
        plt.ylabel("Time (s)")
        plt.title(f"{self.run_name}: Time per Step (Stacked)")
        plt.legend()
        if save_prefix:
            plt.savefig(f"{save_prefix}_timing.png", bbox_inches="tight")
        if show:
            plt.show()

    # ----- export helpers -----
    def export_csv(self, path: str):
        self.to_dataframe().to_csv(path, index=False)

    def export_json(self, path: str):
        self.to_dataframe().to_json(path, orient="records", indent=2)
