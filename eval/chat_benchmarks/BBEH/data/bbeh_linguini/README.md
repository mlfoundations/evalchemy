# BBEH Linguini

This task comes from [Sanches et al. 2024](https://arxiv.org/abs/2409.12126)
where the problems are extracted from the International Linguistic Olympiad
(IOL). The original dataset is available
[here](https://github.com/facebookresearch/linguini). According to the original
work that introduced this dataset, the problems are \emph{"linguistic problems
which require meta-linguistic awareness and deductive reasoning capabilities to
be solved instead of pre-existing language proficiency"}.

We created a subset of the Linguini problems by sampling from four categories of
the Linguini problems, namely *translation*, *fill blanks*, *num to text* and
*text to num*. The original dataset contains questions that require multiple
answers. For example, the *fill blanks* questions have multiple blanks that need
to be filled. We create questions that have a single answer by randomly
selecting one of those blanks and only asking the model to fill that one.
