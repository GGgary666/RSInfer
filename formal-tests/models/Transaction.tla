----------------------------- MODULE Transaction -----------------------------
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS Peers, RemoveOld, LostStop, RollbackFails, Bug
Kinds == {"Prepare", "Pause", "Execute", "Commit", "Resume", "Abort"}
Phases == {"idle", "prepared", "paused", "executed", "committed", "resumed", "aborted", "broken"}
VARIABLES pc, decision, phase, effective, label, paused, broken,
          sent, replies, ack, stopConfirmed, oldAlive, recovery, result
vars == <<pc, decision, phase, effective, label, paused, broken,
          sent, replies, ack, stopConfirmed, oldAlive, recovery, result>>

Init == /\ pc = "prepare" /\ decision = "none"
        /\ phase = [p \in Peers |-> "idle"]
        /\ effective = [p \in Peers |-> 0] /\ label = effective
        /\ paused = [p \in Peers |-> FALSE] /\ broken = paused
        /\ sent = {"Prepare"} /\ replies = [k \in Kinds |-> {}] /\ ack = replies
        /\ stopConfirmed = ~RemoveOld /\ oldAlive = RemoveOld
        /\ recovery = FALSE /\ result = "pending"

\* Sent requests and produced replies remain replayable. Never delivering one
\* abstracts loss; arbitrary delivery order abstracts delay and duplication.
DeliverPrepare(p) ==
    /\ "Prepare" \in sent /\ phase[p] = "idle"
    /\ phase' = [phase EXCEPT ![p] = "prepared"]
    /\ replies' = [replies EXCEPT !["Prepare"] = @ \cup {p}]
    /\ UNCHANGED <<pc, decision, effective, label, paused, broken, sent, ack,
                    stopConfirmed, oldAlive, recovery, result>>
DeliverPause(p) ==
    /\ "Pause" \in sent /\ phase[p] = "prepared"
    /\ phase' = [phase EXCEPT ![p] = "paused"]
    /\ paused' = [paused EXCEPT ![p] = TRUE]
    /\ replies' = [replies EXCEPT !["Pause"] = @ \cup {p}]
    /\ UNCHANGED <<pc, decision, effective, label, broken, sent, ack,
                    stopConfirmed, oldAlive, recovery, result>>
DeliverExecute(p) ==
    /\ "Execute" \in sent /\ phase[p] = "paused"
    /\ phase' = [phase EXCEPT ![p] = "executed"]
    /\ effective' = [effective EXCEPT ![p] = 1]
    /\ replies' = [replies EXCEPT !["Execute"] = @ \cup {p}]
    /\ UNCHANGED <<pc, decision, label, paused, broken, sent, ack,
                    stopConfirmed, oldAlive, recovery, result>>
DeliverCommit(p) ==
    /\ "Commit" \in sent /\ phase[p] = "executed"
    /\ phase' = [phase EXCEPT ![p] = "committed"]
    /\ label' = [label EXCEPT ![p] = 1]
    /\ paused' = [paused EXCEPT ![p] = (Bug # "EarlyResume")]
    /\ replies' = [replies EXCEPT !["Commit"] = @ \cup {p}]
    /\ UNCHANGED <<pc, decision, effective, broken, sent, ack,
                    stopConfirmed, oldAlive, recovery, result>>
DeliverResume(p) ==
    /\ "Resume" \in sent /\ phase[p] = "committed"
    /\ phase' = [phase EXCEPT ![p] = "resumed"]
    /\ paused' = [paused EXCEPT ![p] = FALSE]
    /\ replies' = [replies EXCEPT !["Resume"] = @ \cup {p}]
    /\ UNCHANGED <<pc, decision, effective, label, broken, sent, ack,
                    stopConfirmed, oldAlive, recovery, result>>
DeliverAbort(p) ==
    /\ "Abort" \in sent /\ phase[p] \notin {"committed", "resumed", "aborted", "broken"}
    /\ LET bad == p \in RollbackFails /\ phase[p] # "idle"
           hide == bad /\ Bug = "SwallowRollback"
       IN /\ phase' = [phase EXCEPT ![p] = IF bad /\ ~hide THEN "broken" ELSE "aborted"]
          /\ broken' = [broken EXCEPT ![p] = bad]
          /\ effective' = [effective EXCEPT ![p] = IF bad THEN @ ELSE 0]
          /\ paused' = [paused EXCEPT ![p] = (bad /\ ~hide)]
          /\ replies' = [replies EXCEPT !["Abort"] = IF bad /\ ~hide THEN @ ELSE @ \cup {p}]
    /\ UNCHANGED <<pc, decision, label, sent, ack, stopConfirmed, oldAlive, recovery, result>>

Observe(k, p) ==
    /\ p \in replies[k] /\ p \notin ack[k]
    /\ ack' = [ack EXCEPT ![k] = @ \cup {p}]
    /\ UNCHANGED <<pc, decision, phase, effective, label, paused, broken, sent, replies,
                    stopConfirmed, oldAlive, recovery, result>>
Advance(from, to, kind, nextKind) ==
    /\ pc = from /\ ack[kind] = Peers
    /\ pc' = to /\ sent' = sent \cup {nextKind}
    /\ UNCHANGED <<decision, phase, effective, label, paused, broken, replies, ack,
                    stopConfirmed, oldAlive, recovery, result>>
Executed ==
    /\ pc = "execute" /\ ack["Execute"] = Peers /\ pc' = "stop"
    /\ UNCHANGED <<decision, phase, effective, label, paused, broken, sent, replies, ack,
                    stopConfirmed, oldAlive, recovery, result>>
StopOld ==
    /\ pc = "stop"
    /\ LET unsafe == RemoveOld /\ LostStop
           failed == unsafe /\ Bug # "LostIsStopped"
       IN /\ pc' = IF failed THEN "abort" ELSE "decide"
          /\ decision' = IF failed THEN "abort" ELSE decision
          /\ sent' = IF failed THEN sent \cup {"Abort"} ELSE sent
          /\ recovery' = failed
          /\ stopConfirmed' = ~failed
          /\ oldAlive' = unsafe
    /\ UNCHANGED <<phase, effective, label, paused, broken, replies, ack, result>>
DecideCommit ==
    /\ pc = "decide" /\ decision = "none"
    /\ decision' = "commit" /\ pc' = "commit"
    /\ UNCHANGED <<phase, effective, label, paused, broken, sent, replies, ack,
                    stopConfirmed, oldAlive, recovery, result>>
SendCommit ==
    /\ (pc = "commit" \/ (pc = "decide" /\ Bug = "VolatileDecision"))
    /\ "Commit" \notin sent /\ sent' = sent \cup {"Commit"}
    /\ UNCHANGED <<pc, decision, phase, effective, label, paused, broken, replies, ack,
                    stopConfirmed, oldAlive, recovery, result>>
FailBeforeDecision ==
    /\ decision = "none" /\ pc \notin {"abort", "done"}
    /\ decision' = "abort" /\ pc' = "abort" /\ sent' = sent \cup {"Abort"}
    /\ UNCHANGED <<phase, effective, label, paused, broken, replies, ack,
                    stopConfirmed, oldAlive, recovery, result>>
FinishCommit ==
    /\ pc = "resume" /\ ack["Resume"] = Peers
    /\ pc' = "done" /\ result' = "committed"
    /\ UNCHANGED <<decision, phase, effective, label, paused, broken, sent, replies, ack,
                    stopConfirmed, oldAlive, recovery>>
FinishAbort ==
    /\ pc = "abort" /\ ack["Abort"] = Peers
    /\ pc' = "done" /\ result' = IF recovery THEN "failed" ELSE "aborted"
    /\ UNCHANGED <<decision, phase, effective, label, paused, broken, sent, replies, ack,
                    stopConfirmed, oldAlive, recovery>>
Timeout ==
    /\ pc \in {"abort", "commit", "resume"}
    /\ pc' = "done" /\ result' = "failed" /\ recovery' = (Bug # "NoFreeze")
    /\ UNCHANGED <<decision, phase, effective, label, paused, broken, sent, replies, ack,
                    stopConfirmed, oldAlive>>
UnsafeExpiry(p) ==
    /\ Bug = "UnilateralRollback" /\ phase[p] = "executed"
    /\ phase' = [phase EXCEPT ![p] = "aborted"]
    /\ effective' = [effective EXCEPT ![p] = 0]
    /\ paused' = [paused EXCEPT ![p] = FALSE]
    /\ UNCHANGED <<pc, decision, label, broken, sent, replies, ack,
                    stopConfirmed, oldAlive, recovery, result>>

Next == \/ \E p \in Peers: DeliverPrepare(p) \/ DeliverPause(p) \/ DeliverExecute(p)
                         \/ DeliverCommit(p) \/ DeliverResume(p) \/ DeliverAbort(p) \/ UnsafeExpiry(p)
        \/ \E k \in Kinds, p \in Peers: Observe(k, p)
        \/ Advance("prepare", "pause", "Prepare", "Pause")
        \/ Advance("pause", "execute", "Pause", "Execute")
        \/ Advance("commit", "resume", "Commit", "Resume")
        \/ Executed \/ StopOld \/ DecideCommit \/ SendCommit \/ FailBeforeDecision
        \/ FinishCommit \/ FinishAbort \/ Timeout

Spec == Init /\ [][Next]_vars
TypeOK == /\ phase \in [Peers -> Phases] /\ effective \in [Peers -> {0, 1}]
          /\ label \in [Peers -> {0, 1}] /\ paused \in [Peers -> BOOLEAN]
          /\ broken \in [Peers -> BOOLEAN] /\ sent \subseteq Kinds
          /\ ack \in [Kinds -> SUBSET Peers] /\ replies \in [Kinds -> SUBSET Peers]
          /\ decision \in {"none", "commit", "abort"}
          /\ result \in {"pending", "committed", "aborted", "failed"}
          /\ recovery \in BOOLEAN /\ oldAlive \in BOOLEAN /\ stopConfirmed \in BOOLEAN
DurableBeforeCommit == "Commit" \in sent => decision = "commit"
CommitCompatible == decision = "commit" => \A p \in Peers: phase[p] \in {"executed", "committed", "resumed"}
CommitBarrier == \A p \in Peers: (~paused[p] /\ effective[p] = 1) => ack["Commit"] = Peers
StopEvidence == stopConfirmed => ~oldAlive
RollbackPaused == \A p \in Peers: broken[p] => paused[p]
FailureFreezes == result = "failed" => recovery
HonestCommit == result = "committed" =>
    /\ decision = "commit" /\ stopConfirmed /\ ~oldAlive
    /\ \A p \in Peers: label[p] = 1 /\ effective[p] = 1 /\ ~paused[p]
HonestAbort == result = "aborted" =>
    /\ decision = "abort" /\ ~recovery
    /\ \A p \in Peers: label[p] = 0 /\ effective[p] = 0 /\ ~broken[p]
\* Reachability oracles: deliberately false invariants must produce witnesses.
NeverCommit == result # "committed"
NeverAbort == result # "aborted"
NeverRecovery == ~recovery
=============================================================================
