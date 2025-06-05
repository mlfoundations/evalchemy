# BBEH Boolean Expressions

This task requires determining the truth value of a statement that is composed
of logical operands such as *True* and *False* as well as other textual or
mathematical statements that evaluate to True or False. To create this task, we
first randomly create expressions containing only True and False operands and
three logical operators: **and**, **or**, and **not**. We create this in a
bottom-up fashion where we generate smaller sub-expressions and then combine
them with logical operators. Once a large enough expression is created, we
replace some of the True and False operands with statements that evaluate to
True or False. These could be mathematical expressions such as *24 - 2 is
greater than 48 / 2* (which evaluates to False) or textual statements such as
*The capital of Canada is Ottawa* (which evaluates to True). In both cases, we
select these statements from a predefined set. While determining the truth value
of each of these statements in isolation may be easy for many models, including
these statements makes it more difficult for models; otherwise, they can simply
solve the problem by generating a single line of python code.

We generate five expressions using the approach outlined above, four of which
evaluate to False and one of which evaluate to True. The job of the model is
then to find the expression that evaluates to True. Since this is a five-way
question, the random chance accuracy is 20%.
