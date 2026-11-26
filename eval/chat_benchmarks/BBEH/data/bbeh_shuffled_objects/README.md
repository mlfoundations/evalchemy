# BBEH Shuffled Objects

The original task in BBH is as follows: there are N people each assigned to an
object/person (e.g., a dance partner, a book, a color, etc.). For example, Alice
has a green book, Bob has a red book, etc. Then, there are multiple switch
operations where pairs of people switch together what they are assigned to
(e.g., Alice and Bob switch their books). At the end, one needs to predict the
object/person assigned to one of the N people (e.g., at the end, what color is
the book that Bob has?).

We created two variants of this problem. In the first variant, we keep
everything the same except that we add switch actions that have no effect. For
example, we add *Then, Person1 and Person2 switch their books. Then, Person2 and
Person1 switch their books*. We add many of these no-effect operations so that
the problem becomes a long-context reasoning problem.

The second variant extends the first variant, in which we assign names to some
of the switch actions as they occur and use those names later. For example, the
first time *Person1 switches with Person2* occurs, we replace the text with
*Person1 switches with Person2 (let's call this Action K)*, and the next time
the same switch happens, with some probability we replace the text with
*action K repeats*. Given the long-context nature of the problem, the model
requires to have the ability to remember information from many steps ago to be
able to identify what that action corresponded to.
