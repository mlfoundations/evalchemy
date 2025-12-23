# BBEH BoardgameQA

[BoardgameQA](https://arxiv.org/abs/2306.07934) is a benchmark where given a
defeasible theory (a set of input facts, possibly contradictory rules, and
preferences over the rules), and a question about that theory, the task is to
do multi-hop reasoning and conflict resolution over the input theory to answer
the question. The final answer to the question is either `proved` (if the
statement in the question derives from the theory), `disproved` (if the
negation of the statement in the question derives from the theory), or
`unknown` (if neither the statement in the questions nor its negation derives
from the theory). With three labels per question, a random baseline has an
accuracy of ~33.3\%. Conflicts may arise when two rules such as:

    R1: a implies c
    R2: b implies not c

are both activated leading to different beliefs about the truth value of the
variable c. However, preferences over the rules is provided in the input
question and in the case of conflicts, the derivation from the rule with the
higher preference must be concluded (e.g., if R1 is preferred over R2 and they
both apply, then we conclude c is true).
