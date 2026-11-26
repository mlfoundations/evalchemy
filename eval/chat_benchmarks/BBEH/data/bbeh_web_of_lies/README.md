# BBEH Web of Lies

In this task, whether a specific person P1 tells the truth of lies is provided
as input. Then, for other people, it is specified what they say about the truth
value of some other person. This forms a chain-like structure that can be
started from P1 and continued to find whether each of the people tells the
truth or lies.

We used two different variants for this task. The first variant comes from the
*Web of Lies V2* from [LiveBench](https://arxiv.org/abs/2406.19314). In this
variant, complexity has been added to the task by specifying where each person
is, and then having sentences such as *The person at the cafe says the person
at the zoo lies*. The second version is created by us. In this version, we add
cyclic cases whose truth value remains unknown, but one can still infer
something about them and continue the chain. For example, consider a cyclic case
such as *Person1 says Person2 tells the truth. Person2 says Person1 tells the
truth.* In this case, we cannot determine whether Person1 or Person2 tell the
truth or lie (so their truthfulness remains unknown). However, if we have
another sentence *Person3 says either both Person1 and Person2 lie or both tell
the truth*, we can determine that Person3 tells the truth. In both variants of
the problems, we ask about the truthfulness of three of the people in the chain,
so the random chance performance for the LiveBench subset is 1/8 since the
truthfulness of each of the three people can be either yes or no, and 1/27 for
our new set given that the values can also be unknown.
