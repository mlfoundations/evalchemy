# BBEH Zebra Puzzles

Zebra puzzles, also known as Einstein puzzles, are verbal descriptions of
entities and properties that partially populate a grid linking entities to their
properties. The description may also include constraints on these properties,
such that it is possible to deduce the other entity-property links. Following
the approach [this work](https://arxiv.org/abs/2409.10502), we generate
square-grid Zebra puzzles of size 5x5, 6x6, 7x7, and 8x8. We add distracting
clues to puzzles of size 5, 6, and 7 to make them more challenging, but do not
add them to puzzles of size 8 to avoid keeping the context size too large. To
simplify evaluation, the questions ask for the position of one of the n people
in the n x n puzzles, so the random chance performance for a n x n puzzle is
1 / n.
