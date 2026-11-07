----------------------------- MODULE Participant -----------------------------
EXTENDS Naturals, FiniteSets, TLC
CONSTANTS Txns, ExecuteFails, AbortFails, Bug
VARIABLES current, phase, finished, closed, value, identity, before, beforeIdentity,
          paused, undoFailed, execs, aborts, event
vars == <<current, phase, finished, closed, value, identity, before, beforeIdentity,
          paused, undoFailed, execs, aborts, event>>
Init == /\ current = 0 /\ phase = "idle"
        /\ finished = [t \in Txns |-> "none"] /\ closed = {}
        /\ value = 0 /\ identity = 0 /\ before = 0 /\ beforeIdentity = 0
        /\ paused = FALSE /\ undoFailed = FALSE
        /\ execs = [t \in Txns |-> 0] /\ aborts = execs
        /\ event = [op |-> "Init", tx |-> 0]

Prepare(t) ==
    /\ event' = [op |-> "Prepare", tx |-> t]
    /\ IF current = 0 /\ finished[t] = "none"
       THEN /\ current' = t /\ phase' = "prepared"
            /\ before' = value /\ beforeIdentity' = identity
       ELSE UNCHANGED <<current, phase, before, beforeIdentity>>
    /\ UNCHANGED <<finished, closed, value, identity, paused, undoFailed, execs, aborts>>
Pause(t) ==
    /\ event' = [op |-> "Pause", tx |-> t]
    /\ IF current = t /\ (phase = "prepared" \/ (phase = "undone" /\ Bug = "RearmUndone"))
       THEN /\ phase' = "paused" /\ paused' = TRUE
       ELSE UNCHANGED <<phase, paused>>
    /\ UNCHANGED <<current, finished, closed, value, identity, before, beforeIdentity, undoFailed, execs, aborts>>
Execute(t) ==
    /\ event' = [op |-> "Execute", tx |-> t]
    /\ IF current = t /\ phase = "paused"
       THEN /\ execs' = [execs EXCEPT ![t] = @ + 1]
            /\ phase' = IF ExecuteFails THEN "undone" ELSE "executed"
            /\ value' = IF ExecuteFails /\ ~AbortFails THEN before ELSE t
            /\ undoFailed' = (ExecuteFails /\ AbortFails)
            /\ aborts' = IF ExecuteFails THEN [aborts EXCEPT ![t] = @ + 1] ELSE aborts
       ELSE UNCHANGED <<execs, phase, value, undoFailed, aborts>>
    /\ UNCHANGED <<current, finished, closed, identity, before, beforeIdentity, paused>>
Commit(t) ==
    /\ event' = [op |-> "Commit", tx |-> t]
    /\ IF current = t /\ phase = "executed"
       THEN /\ identity' = t /\ phase' = "committed"
       ELSE UNCHANGED <<identity, phase>>
    /\ UNCHANGED <<current, finished, closed, value, before, beforeIdentity, paused, undoFailed, execs, aborts>>
Resume(t) ==
    /\ event' = [op |-> "Resume", tx |-> t]
    /\ IF current = t /\ phase = "committed"
       THEN /\ current' = 0 /\ phase' = "idle" /\ paused' = FALSE
            /\ finished' = [finished EXCEPT ![t] = "commit"] /\ closed' = closed \cup {t}
       ELSE UNCHANGED <<current, phase, paused, finished, closed>>
    /\ UNCHANGED <<value, identity, before, beforeIdentity, undoFailed, execs, aborts>>
Abort(t) ==
    /\ event' = [op |-> "Abort", tx |-> t]
    /\ IF finished[t] # "none" \/ (current = t /\ phase = "committed")
       THEN UNCHANGED <<current, phase, paused, finished, closed, value, undoFailed, aborts>>
       ELSE IF current # t
            THEN /\ finished' = IF Bug = "ForgetAbort" THEN finished ELSE [finished EXCEPT ![t] = "abort"]
                 /\ closed' = closed \cup {t}
                 /\ UNCHANGED <<current, phase, paused, value, undoFailed, aborts>>
            ELSE LET bad == undoFailed \/ (phase # "undone" /\ AbortFails)
                 IN /\ aborts' = IF phase = "undone" THEN aborts ELSE [aborts EXCEPT ![t] = @ + 1]
                    /\ undoFailed' = bad
                    /\ IF bad
                       THEN /\ paused' = (Bug # "ResumeFailedUndo") /\ phase' = "undone"
                            /\ UNCHANGED <<current, finished, closed, value>>
                       ELSE /\ current' = 0 /\ phase' = "idle" /\ paused' = FALSE
                            /\ value' = before
                            /\ finished' = [finished EXCEPT ![t] = "abort"] /\ closed' = closed \cup {t}
    /\ UNCHANGED <<identity, before, beforeIdentity, execs>>

Next == \E t \in Txns: Prepare(t) \/ Pause(t) \/ Execute(t) \/ Commit(t) \/ Resume(t) \/ Abort(t)
Spec == Init /\ [][Next]_vars
TypeOK == /\ current \in Txns \cup {0} /\ identity \in Txns \cup {0} /\ value \in Txns \cup {0}
          /\ phase \in {"idle", "prepared", "paused", "executed", "undone", "committed"}
          /\ finished \in [Txns -> {"none", "commit", "abort"}] /\ closed \subseteq Txns
          /\ paused \in BOOLEAN /\ undoFailed \in BOOLEAN
ClosedNotActive == current = 0 \/ current \notin closed
RollbackPaused == undoFailed => paused
ServingMatchesEffect == ~paused => value = identity
CommitHasEffect == phase = "committed" => value = identity /\ identity = current
AtMostOnceExecute == \A t \in Txns: execs[t] <= 1
AtMostOnceAbort == \A t \in Txns: aborts[t] <= 1
NeverCommitted == \A t \in Txns: finished[t] # "commit"
NeverExecutedAbort == ~\E t \in Txns: finished[t] = "abort" /\ execs[t] = 1
NeverUndoFailure == ~undoFailed
NeverTwoCommits == Cardinality({t \in Txns: finished[t] = "commit"}) < 2
NeverCrossAbort == ~(current # 0 /\ \E t \in closed: t # current)
=============================================================================
