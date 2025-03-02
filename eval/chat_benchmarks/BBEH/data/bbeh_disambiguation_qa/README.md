# BBEH DisambiguationQA

This task introduces a more challenging adaptation of the original
Disambiguation task in BBH. The objective is to accurately determine the
referents of ambiguous pronouns in complex sentences, or to explicitly identify
instances of unresolvable ambiguity by responding 'ambiguous'. To enhance the
task difficulty and complexity, we constructed a dataset of 120 novel examples
that are longer than those in BBH, require more referent disambiguation, and
each question contains more options so the random chance performance is lower.
These examples were constructed either by creating entirely new sentences or
combining existing BBH instances. Ten annotators (all of them the authors of the
paper) were tasked with creating these examples, each comprising a potentially
ambiguous sentence, a single correct resolution statement, and several
distractor options for a multiple-choice format. To ensure data quality, each
example underwent a two-stage verification process. First, a separate
annotator independently evaluated the correctness of the resolution.
Discrepancies were then resolved through a third-party adjudicator or
collaborative refinement by all three annotators. In cases where consensus
could not be reached, the annotators jointly revised the example to achieve
clarity and accuracy. This rigorous process resulted in 25 examples requiring
modification.
