----------------------- MODULE SequentialTransactions -----------------------
EXTENDS Naturals, FiniteSets, TLC
CONSTANTS Peers, RollbackFails, RestartPeers, Bug
Txns == {1, 2}
Kinds == {"Prepare", "Pause", "Execute", "Commit", "Resume", "Abort"}
VARIABLE s
vars == <<s>>
Msg(t, k, p) == <<t, k, p>>
Broadcast(t, k) == {Msg(t, k, p): p \in Peers}
AllAck(k) == Broadcast(s.tx, k) \subseteq s.acks

Init == s = [tx |-> 1, pc |-> "prepare", recovery |-> FALSE,
    generation |-> [p \in Peers |-> 1], destination |-> [t \in Txns |-> [p \in Peers |-> 1]],
    current |-> [p \in Peers |-> 0], phase |-> [p \in Peers |-> "idle"],
    value |-> [p \in Peers |-> 0], identity |-> [p \in Peers |-> 0],
    paused |-> [p \in Peers |-> FALSE], broken |-> {},
    closed |-> {}, sent |-> Broadcast(1, "Prepare"), replies |-> {}, acks |-> {},
    result |-> [t \in Txns |-> "pending"], decision |-> [t \in Txns |-> "none"],
    lateDuringSecond |-> FALSE]

DeliverPrepare(t, p) ==
    /\ Msg(t, "Prepare", p) \in s.sent
    /\ (s.destination[t][p] = s.generation[p] \/ Bug = "RouteAcrossIncarnation")
    /\ s.current[p] = 0 /\ <<t, p>> \notin s.closed
    /\ s' = [s EXCEPT !.current[p] = t, !.phase[p] = "prepared",
             !.replies = @ \cup {Msg(t, "Prepare", p)}]
DeliverPause(t, p) ==
    /\ Msg(t, "Pause", p) \in s.sent /\ s.current[p] = t /\ s.phase[p] = "prepared"
    /\ s' = [s EXCEPT !.phase[p] = "paused", !.paused[p] = TRUE,
             !.replies = @ \cup {Msg(t, "Pause", p)}]
DeliverExecute(t, p) ==
    /\ Msg(t, "Execute", p) \in s.sent /\ s.current[p] = t /\ s.phase[p] = "paused"
    /\ s' = [s EXCEPT !.phase[p] = "executed", !.value[p] = t,
             !.replies = @ \cup {Msg(t, "Execute", p)}]
DeliverCommit(t, p) ==
    /\ Msg(t, "Commit", p) \in s.sent /\ s.current[p] = t /\ s.phase[p] = "executed"
    /\ s' = [s EXCEPT !.phase[p] = "committed", !.identity[p] = t,
             !.replies = @ \cup {Msg(t, "Commit", p)}]
DeliverResume(t, p) ==
    /\ Msg(t, "Resume", p) \in s.sent /\ s.current[p] = t /\ s.phase[p] = "committed"
    /\ s' = [s EXCEPT !.current[p] = 0, !.phase[p] = "idle", !.paused[p] = FALSE,
             !.closed = @ \cup {<<t, p>>}, !.replies = @ \cup {Msg(t, "Resume", p)}]
DeliverAbort(p) ==
    /\ Msg(1, "Abort", p) \in s.sent /\ s.current[p] = 1
    /\ s.phase[p] \notin {"broken", "committed"}
    /\ IF p \in RollbackFails
       THEN s' = [s EXCEPT !.phase[p] = "broken", !.broken = @ \cup {p}, !.paused[p] = TRUE]
       ELSE s' = [s EXCEPT !.current[p] = 0, !.phase[p] = "idle", !.paused[p] = FALSE,
                  !.value[p] = 0, !.closed = @ \cup {<<1, p>>},
                  !.replies = @ \cup {Msg(1, "Abort", p)}]

\* Replay a real, previously sent request after the second transaction starts.
\* Normal handlers leave the new transaction alone. The mutant aborts its work.
LateRequest(k, p) ==
    /\ s.tx = 2 /\ s.current[p] = 2 /\ Msg(1, k, p) \in s.sent
    /\ s.destination[1][p] = s.generation[p]
    /\ IF Bug = "OldAbortTouchesNew" /\ k = "Abort"
       THEN s' = [s EXCEPT !.lateDuringSecond = TRUE, !.value[p] = 0,
                  !.current[p] = 0, !.phase[p] = "idle", !.paused[p] = FALSE]
       ELSE s' = [s EXCEPT !.lateDuringSecond = TRUE]
Observe(t, k, p) ==
    /\ Msg(t, k, p) \in s.replies
    /\ LET used == IF Bug = "OldReceiptCountsForNew" /\ s.tx = 2 THEN 2 ELSE t
       IN /\ Msg(used, k, p) \notin s.acks
          /\ s' = [s EXCEPT !.acks = @ \cup {Msg(used, k, p)}]
Advance(from, to, kind, nextKind) ==
    /\ s.pc = from /\ AllAck(kind)
    /\ s' = [s EXCEPT !.pc = to, !.sent = @ \cup Broadcast(s.tx, nextKind)]
Decide ==
    /\ s.pc = "execute" /\ AllAck("Execute")
    /\ IF s.tx = 1
       THEN s' = [s EXCEPT !.pc = "abort", !.decision[1] = "abort",
                  !.sent = @ \cup Broadcast(1, "Abort")]
       ELSE s' = [s EXCEPT !.pc = "commit", !.decision[2] = "commit"]
SendCommit ==
    /\ s.pc = "commit" /\ s.decision[s.tx] = "commit"
    /\ ~(Broadcast(s.tx, "Commit") \subseteq s.sent)
    /\ s' = [s EXCEPT !.sent = @ \cup Broadcast(s.tx, "Commit")]
Finish ==
    /\ s.pc \in {"abort", "resume"}
    /\ AllAck(IF s.pc = "abort" THEN "Abort" ELSE "Resume")
    /\ s' = [s EXCEPT !.pc = "done", !.result[s.tx] = IF s.pc = "abort" THEN "aborted" ELSE "committed"]
Timeout ==
    /\ s.pc \in {"abort", "commit", "resume"}
    /\ s' = [s EXCEPT !.pc = "done", !.result[s.tx] = "failed", !.recovery = TRUE]
NextTransaction ==
    /\ s.tx = 1 /\ s.pc = "done"
    /\ (~s.recovery \/ Bug = "IgnoreRecovery")
    /\ s' = [s EXCEPT !.tx = 2, !.pc = "prepare", !.sent = @ \cup Broadcast(2, "Prepare"),
             !.destination[2] = s.generation]
\* Successful rollback permits replacement before the next transaction.
\* A fresh process loses in-memory tombstones. Requests remain bound to their
\* original endpoint generation; this is a transport assumption, not an extra
\* incarnation field in the Python change messages.
Restart(p) ==
    /\ RestartPeers /\ s.tx = 1 /\ s.pc = "done" /\ s.result[1] = "aborted"
    /\ ~s.recovery /\ s.current[p] = 0 /\ s.generation[p] = 1
    /\ s' = [s EXCEPT !.generation[p] = 2, !.closed = @ \ {<<1, p>>}]

Next == \/ \E t \in Txns, p \in Peers:
              DeliverPrepare(t, p) \/ DeliverPause(t, p) \/ DeliverExecute(t, p)
              \/ DeliverCommit(t, p) \/ DeliverResume(t, p)
        \/ \E p \in Peers: DeliverAbort(p)
        \/ \E k \in Kinds, p \in Peers: LateRequest(k, p)
        \/ \E t \in Txns, k \in Kinds, p \in Peers: Observe(t, k, p)
        \/ Advance("prepare", "pause", "Prepare", "Pause")
        \/ Advance("pause", "execute", "Pause", "Execute")
        \/ Advance("commit", "resume", "Commit", "Resume")
        \/ Decide \/ SendCommit \/ Finish \/ Timeout \/ NextTransaction
        \/ \E p \in Peers: Restart(p)
Spec == Init /\ [][Next]_vars
TypeOK == /\ s.tx \in Txns /\ s.current \in [Peers -> Txns \cup {0}]
          /\ s.value \in [Peers -> {0, 1, 2}] /\ s.identity \in [Peers -> {0, 1, 2}]
          /\ s.paused \in [Peers -> BOOLEAN] /\ s.broken \subseteq Peers
ReceiptHasEvidence == s.acks \subseteq s.replies
NoClosedReactivation == \A p \in Peers: <<s.current[p], p>> \notin s.closed
CompletedFirstNotReopened == s.result[1] = "aborted" => \A p \in Peers: s.current[p] # 1
NewTransactionPreserved == \A p \in Peers:
    (Msg(2, "Prepare", p) \in s.replies /\ <<2, p>> \notin s.closed) => s.current[p] = 2
FailedFirstBlocksSecond == s.result[1] = "failed" => s.tx = 1
RollbackPaused == \A p \in s.broken: s.paused[p]
ServingMatchesEffect == \A p \in Peers: ~s.paused[p] => s.value[p] = s.identity[p]
CommitBarrier == (\E p \in Peers: ~s.paused[p] /\ s.identity[p] = 2) =>
                 Broadcast(2, "Commit") \subseteq s.acks
ExecutedEvidence == s.pc \in {"commit", "resume"} =>
                    \A p \in Peers: s.value[p] = 2
HonestSecondCommit == s.result[2] = "committed" =>
                      \A p \in Peers: s.value[p] = 2 /\ s.identity[p] = 2 /\ ~s.paused[p]
NeverSecondCommit == s.result[2] # "committed"
NeverLateThenCommit == ~(s.lateDuringSecond /\ s.result[2] = "committed")
NeverFailedRollback == s.broken = {}
NeverRestartThenCommit == ~((\E p \in Peers: s.generation[p] = 2) /\ s.result[2] = "committed")
=============================================================================
