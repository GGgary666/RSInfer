#!/usr/bin/env python3
"""Run bounded TLC checks; reject timeouts, syntax errors and unexpected failures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
import urllib.request

ROOT = Path(__file__).resolve().parent
VERSION = "1.7.4"
JAR_SHA256 = "936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88"
URL = f"https://github.com/tlaplus/tlaplus/releases/download/v{VERSION}/tla2tools.jar"
TX_INVARIANTS = ["TypeOK", "DurableBeforeCommit", "CommitCompatible", "CommitBarrier", "StopEvidence",
                 "RollbackPaused", "FailureFreezes", "HonestCommit", "HonestAbort"]
PEER_INVARIANTS = ["TypeOK", "ClosedNotActive", "RollbackPaused", "ServingMatchesEffect", "CommitHasEffect",
                   "AtMostOnceExecute", "AtMostOnceAbort"]
SEQUENCE_INVARIANTS = ["TypeOK", "ReceiptHasEvidence", "NoClosedReactivation", "CompletedFirstNotReopened", "NewTransactionPreserved",
                       "FailedFirstBlocksSecond", "RollbackPaused", "ServingMatchesEffect", "CommitBarrier",
                       "ExecutedEvidence", "HonestSecondCommit"]
INCARNATION_INVARIANTS = ["TypeOK", "SingleLiveProcess", "EveryLiveProcessTracked", "NoStaleStopKillsSuccessor",
                         "NoStaleReportChangesSuccessor", "GenerationNeverGoesBack", "RunningReportMatchesGeneration"]


def cases() -> list[dict[str, Any]]:
    tx = {"Peers": "{1, 2}", "RemoveOld": "FALSE", "LostStop": "FALSE", "RollbackFails": "{}", "Bug": '"None"'}
    peer = {"Txns": "{1, 2}", "ExecuteFails": "FALSE", "AbortFails": "FALSE", "Bug": '"None"'}
    out = []

    def add(name, module, constants, overrides=None, expected=None):
        out.append({"name": name, "module": module, "constants": {**constants, **(overrides or {})},
                    "invariants": {"Transaction": TX_INVARIANTS, "Participant": PEER_INVARIANTS,
                                   "SequentialTransactions": SEQUENCE_INVARIANTS,
                                   "Incarnation": INCARNATION_INVARIANTS}[module],
                    "expected": expected})

    add("transaction", "Transaction", tx)
    add("transaction-stop", "Transaction", tx, {"RemoveOld": "TRUE"})
    add("transaction-lost", "Transaction", tx, {"RemoveOld": "TRUE", "LostStop": "TRUE"})
    add("transaction-rollback", "Transaction", tx, {"RollbackFails": "{2}"})
    add("participant", "Participant", peer)
    add("participant-execute-fails", "Participant", peer, {"ExecuteFails": "TRUE"})
    add("participant-abort-fails", "Participant", peer, {"AbortFails": "TRUE"})
    add("participant-both-fail", "Participant", peer, {"ExecuteFails": "TRUE", "AbortFails": "TRUE"})
    for bug, invariant, extra in [
        ("EarlyResume", "CommitBarrier", {}),
        ("LostIsStopped", "StopEvidence", {"RemoveOld": "TRUE", "LostStop": "TRUE"}),
        ("SwallowRollback", "RollbackPaused", {"RollbackFails": "{2}"}),
        ("NoFreeze", "FailureFreezes", {}),
        ("VolatileDecision", "DurableBeforeCommit", {}),
        ("UnilateralRollback", "CommitCompatible", {}),
    ]:
        add("mutant-" + bug, "Transaction", tx, {**extra, "Bug": json.dumps(bug)}, invariant)
    for bug, invariant, extra in [
        ("ForgetAbort", "ClosedNotActive", {}),
        ("RearmUndone", "AtMostOnceExecute", {"ExecuteFails": "TRUE"}),
        ("ResumeFailedUndo", "RollbackPaused", {"AbortFails": "TRUE"}),
    ]:
        add("mutant-" + bug, "Participant", peer, {**extra, "Bug": json.dumps(bug)}, invariant)
    for invariant in ["NeverCommit", "NeverAbort", "NeverRecovery"]:
        add("witness-" + invariant, "Transaction", tx, expected=invariant)
    for invariant, extra in [("NeverCommitted", {}), ("NeverExecutedAbort", {}),
                              ("NeverTwoCommits", {}), ("NeverCrossAbort", {}),
                              ("NeverUndoFailure", {"AbortFails": "TRUE"})]:
        add("witness-" + invariant, "Participant", peer, extra, invariant)
    sequence = {"Peers": "{1, 2}", "RollbackFails": "{}", "RestartPeers": "FALSE", "Bug": '"None"'}
    add("sequence", "SequentialTransactions", sequence)
    add("sequence-rollback-fails", "SequentialTransactions", sequence, {"RollbackFails": "{2}"})
    add("sequence-restart", "SequentialTransactions", sequence, {"RestartPeers": "TRUE"})
    for bug, invariant, extra in [
        ("OldAbortTouchesNew", "NewTransactionPreserved", {}),
        ("OldReceiptCountsForNew", "ReceiptHasEvidence", {}),
        ("IgnoreRecovery", "FailedFirstBlocksSecond", {"RollbackFails": "{2}"}),
        ("RouteAcrossIncarnation", "CompletedFirstNotReopened", {"RestartPeers": "TRUE"}),
    ]:
        add("mutant-" + bug, "SequentialTransactions", sequence, {**extra, "Bug": json.dumps(bug)}, invariant)
    for invariant, extra in [("NeverSecondCommit", {}), ("NeverLateThenCommit", {}),
                              ("NeverRestartThenCommit", {"RestartPeers": "TRUE"}),
                              ("NeverFailedRollback", {"RollbackFails": "{2}"})]:
        add("witness-" + invariant, "SequentialTransactions", sequence, extra, invariant)
    incarnation = {"Bug": '"None"'}
    add("incarnation", "Incarnation", incarnation)
    for bug, invariant in [("NoLifecycleLock", "SingleLiveProcess"),
                           ("IgnoreStopGeneration", "NoStaleStopKillsSuccessor"),
                           ("AcceptStaleReport", "NoStaleReportChangesSuccessor"),
                           ("AllowOlderSpawn", "GenerationNeverGoesBack")]:
        add("mutant-" + bug, "Incarnation", incarnation, {"Bug": json.dumps(bug)}, invariant)
    for invariant in ["NeverReplacementRunning", "NeverSurvivedStaleStop"]:
        add("witness-" + invariant, "Incarnation", incarnation, expected=invariant)
    return out


def tool(path: Path, bootstrap: bool) -> Path:
    if not path.exists():
        if not bootstrap:
            raise RuntimeError("TLC jar missing; run with --bootstrap or provide --jar")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != JAR_SHA256:
            raise RuntimeError("downloaded TLC jar checksum mismatch")
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != JAR_SHA256:
        raise RuntimeError("TLC jar checksum mismatch; expected pinned v" + VERSION)
    return path.resolve()


def export_trace(stdout: str, path: Path) -> None:
    """Export TLC 1.7.4 safety traces; retain raw states, parse only scalar observables.

    This is deliberately not a general TLA+ value parser. Functions/sets remain
    raw text. Replay consumes the Participant model's scalar fields and event.
    """
    blocks = re.split(r"(?m)^State \d+:.*\n", stdout)[1:]
    if not blocks:
        raise RuntimeError("TLC reported a violation without a state trace")
    states = []
    for block in blocks:
        values = {}
        for name, literal in re.findall(r"(?m)^/\\ (\w+) = (.*)$", block):
            if literal in ("TRUE", "FALSE"):
                values[name] = literal == "TRUE"
            elif re.fullmatch(r"\d+", literal):
                values[name] = int(literal)
            elif re.fullmatch(r'"[^"\\]*"', literal):
                values[name] = json.loads(literal)
        event = re.search(r'/\\ event = \[op \|-> "(\w+)", tx \|-> (\d+)\]', block)
        states.append({"values": values, "event": {"op": event[1], "tx": int(event[2])} if event else None,
                       "raw": block.strip()})
    if len(states) < 2:
        raise RuntimeError("expected a nontrivial counterexample/witness")
    path.write_text(json.dumps({"states": states}, indent=2) + "\n")


def check(case: dict[str, Any], jar: Path, java: str, timeout: int, output: Path) -> dict[str, Any]:
    dest = output / case["name"]
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "result.json").unlink(missing_ok=True)
    model = case["module"]
    shutil.copyfile(ROOT / "models" / (model + ".tla"), dest / (model + ".tla"))
    invariants = list(case["invariants"])
    if case["expected"] and case["expected"] not in invariants:
        invariants.append(case["expected"])
    config = "SPECIFICATION Spec\nCHECK_DEADLOCK FALSE\nCONSTANTS\n"
    config += "".join(f"  {k} = {v}\n" for k, v in case["constants"].items())
    config += "INVARIANTS\n" + "".join(f"  {x}\n" for x in invariants)
    (dest / (model + ".cfg")).write_text(config)
    trace = dest / "trace.json"
    trace.unlink(missing_ok=True)  # never accept a trace left by an earlier run
    cmd = [java, "-Xmx1g", "-XX:+UseParallelGC", "-cp", str(jar), "tlc2.TLC", "-workers", "1",
           "-cleanup", "-metadir", str(dest / "states"),
           "-config", model + ".cfg", model]
    try:
        result = subprocess.run(cmd, cwd=dest, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=timeout)
    except subprocess.TimeoutExpired as e:
        (dest / "tlc.log").write_bytes(e.stdout or b"")
        raise RuntimeError(f"{case['name']}: timeout; verification is incomplete") from e
    (dest / "tlc.log").write_text(result.stdout)
    violation = re.search(r"Invariant (\w+) is violated", result.stdout)
    if violation:
        export_trace(result.stdout, trace)
    expected = case["expected"]
    if expected:
        ok = result.returncode != 0 and violation is not None and violation[1] == expected and trace.exists()
    else:
        ok = result.returncode == 0 and "Model checking completed. No error has been found." in result.stdout
    if not ok:
        raise RuntimeError(f"{case['name']}: unexpected TLC result (exit {result.returncode})\n{result.stdout[-4500:]}")
    counts = re.search(r"([\d,]+) states generated, ([\d,]+) distinct states found", result.stdout)
    summary = {**case, "outcome": "counterexample" if expected else "checked",
               "model_sha256": hashlib.sha256((dest / (model + ".tla")).read_bytes()).hexdigest(),
               "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest() if trace.exists() else None,
               "tlc_version": VERSION,
               "distinct_states": int(counts[2].replace(",", "")) if counts else None}
    (dest / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{case['name']}: {summary['outcome']} {expected or ''} ({summary['distinct_states']} states)", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", action="store_true", help="download the checksum-pinned TLC jar if missing")
    parser.add_argument("--jar", type=Path, default=ROOT / ".tools" / "tla2tools.jar")
    parser.add_argument("--java", default=os.environ.get("JAVA", "java"))
    parser.add_argument("--timeout", type=int, default=180, help="seconds per model; a timeout is a failure")
    parser.add_argument("--case", action="append", dest="names", help="run only this named case; repeatable")
    parser.add_argument("--output", type=Path, default=ROOT / ".runs")
    parser.add_argument("--replay", action="store_true", help="also run Python tests with this interpreter")
    args = parser.parse_args()
    selected = [c for c in cases() if args.names is None or c["name"] in args.names]
    if not selected or (args.names and set(args.names) - {c["name"] for c in selected}):
        parser.error("unknown case name")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").unlink(missing_ok=True)
    jar = tool(args.jar, args.bootstrap)
    summaries = [check(c, jar, args.java, args.timeout, output) for c in selected]
    (output / "report.json").write_text(json.dumps(summaries, indent=2) + "\n")
    if args.replay:
        return subprocess.call([sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q"],
                               cwd=ROOT.parent, env={**os.environ, "FORMAL_RUNS": str(output)})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
