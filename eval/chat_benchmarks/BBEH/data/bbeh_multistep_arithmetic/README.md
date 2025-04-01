# BBEH Multi-Step Arithmetic

This task introduces new arithmetic operators. An example of such an operator
is as follows:

    a >< b equals (a - b) if a * b > 0; otherwise, it equals a + b

Some of the operations can be defined based on the other new operations. For
example we may have:

    a ; b equals (a >< b) if a + b > 0; otherwise, it equals a - b

We also define a form of composing multiple operations as follows: a op1 op2 b
denotes (a op1 b) op2 b; for example, 4 +* -5 means (4 +~ 5) * -5 and 4 *++ 5
means (4 * 5) ++ 5.

Then we sample random arithmetic expressions involving the above operations. An
example expression is:

    (1 @*+ 4) <>+[] (-4 *<>* -1)

(although our expressions are longer), with @, <>, and [] being new operations.
The job of the model is to compute the value of the expression. Being able to
compute these expressions requires expanding the expressions and making a long
list of computations correctly.
