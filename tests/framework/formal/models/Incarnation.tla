----------------------------- MODULE Incarnation -----------------------------
EXTENDS Naturals, FiniteSets, TLC
CONSTANT Bug
Requests == {1, 2}
VARIABLE s
vars == <<s>>
Init == s = [live |-> {0}, table |-> 0, pending |-> {}, done |-> {}, lock |-> 0,
             supervisorGeneration |-> 1, supervisorState |-> "RUNNING", reportGeneration |-> 1,
             everNew |-> FALSE, staleKilled |-> FALSE, staleChanged |-> FALSE, staleSeen |-> FALSE]
Generation(child) == IF child = 0 THEN 1 ELSE 2

StopOld ==
    /\ s.live = {0}
    /\ s' = [s EXCEPT !.live = {}, !.supervisorState = "STOPPED"]
RequestReplacement ==
    /\ s.supervisorGeneration = 1 /\ s.live = {}
    /\ s' = [s EXCEPT !.supervisorGeneration = 2, !.supervisorState = "SPAWNING"]
\* Two repeated Spawn requests can overlap at create_subprocess_exec's await.
\* With the per-ProcId lifecycle lock, the second request cannot pass the checks.
BeginSpawn(r) ==
    /\ s.supervisorGeneration = 2 /\ s.table = 0 /\ s.live = {}
    /\ r \notin s.pending /\ r \notin s.done
    /\ (s.lock = 0 \/ Bug = "NoLifecycleLock")
    /\ s' = [s EXCEPT !.pending = @ \cup {r}, !.lock = r]
CompleteSpawn(r) ==
    /\ r \in s.pending
    /\ s' = [s EXCEPT !.pending = @ \ {r}, !.done = @ \cup {r}, !.lock = 0,
             !.live = @ \cup {r}, !.table = r, !.everNew = TRUE]
CheckIn ==
    /\ s.table \in s.live /\ Generation(s.table) = 2 /\ s.supervisorState = "SPAWNING"
    /\ s' = [s EXCEPT !.supervisorState = "RUNNING", !.reportGeneration = 2]
LateStop ==
    /\ s.everNew
    /\ IF Bug = "IgnoreStopGeneration"
       THEN s' = [s EXCEPT !.live = {}, !.staleKilled = (s.live # {}), !.staleSeen = TRUE]
       ELSE s' = [s EXCEPT !.staleSeen = TRUE]
LateExit ==
    /\ s.supervisorGeneration = 2
    /\ IF Bug = "AcceptStaleReport"
       THEN s' = [s EXCEPT !.supervisorState = "LOST", !.staleChanged = TRUE]
       ELSE UNCHANGED s
StopNew ==
    /\ s.everNew /\ s.pending = {} /\ s.live # {}
    /\ s' = [s EXCEPT !.live = {}, !.supervisorState = "STOPPED"]
LateSpawn ==
    /\ s.everNew /\ s.live = {} /\ s.pending = {}
    /\ IF Bug = "AllowOlderSpawn"
       THEN s' = [s EXCEPT !.table = 0, !.live = {0}]
       ELSE UNCHANGED s

Next == StopOld \/ RequestReplacement \/ CheckIn \/ LateStop \/ LateExit \/ StopNew \/ LateSpawn
        \/ \E r \in Requests: BeginSpawn(r) \/ CompleteSpawn(r)
Spec == Init /\ [][Next]_vars
TypeOK == /\ s.live \subseteq {0, 1, 2} /\ s.table \in {0, 1, 2}
          /\ s.pending \subseteq Requests /\ s.done \subseteq Requests
          /\ s.supervisorGeneration \in {1, 2}
SingleLiveProcess == Cardinality(s.live) <= 1
EveryLiveProcessTracked == s.live \subseteq {s.table}
NoStaleStopKillsSuccessor == ~s.staleKilled
NoStaleReportChangesSuccessor == ~s.staleChanged
GenerationNeverGoesBack == s.everNew => s.table # 0
RunningReportMatchesGeneration == s.supervisorState = "RUNNING" =>
                                  s.reportGeneration = s.supervisorGeneration
NeverReplacementRunning == ~(s.everNew /\ s.supervisorState = "RUNNING")
NeverSurvivedStaleStop == ~(s.staleSeen /\ s.supervisorState = "RUNNING" /\ s.live # {})
=============================================================================
