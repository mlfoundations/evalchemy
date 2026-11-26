# BBEH Time Arithmetic

This task is based on the Time Arithmetic subset of the
[Test of Time (ToT)](https://arxiv.org/abs/2406.09170) benchmark. The original
subset, available [here](https://huggingface.co/datasets/baharef/ToT), contains
various questions about understanding, computations over, comparisons, and
conversions of dates and times. There are also trick questions which may require
extra thinking. The dataset also contains some scheduling problems, but we
removed that subset given that we have an entire task (Temporal Sequence)
dedicated to it.

We created a compositional version of the ToT Time Arithmetic dataset as
follows. Let Q1 and Q2 be two questions from the original dataset, where the
answer to the Q1 is A1 (A1 being a number) and let A2 be a number that is used
in Q2. Then, we create a compositional question as follows:

    Let the answer to Q1 be X.
    Q1: <Text of the question>.
    Let X' = X + (A2 - A1). Use this value to solve Q2.
    Q2: <Text of the question with A2 replaced with "X'">.
