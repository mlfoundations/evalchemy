# BBEH Spatial Reasoning

This task is mainly based on the problems in
[SpacialLLMEval](https://arxiv.org/abs/2310.14540). The original dataset was
obtained from [here](https://github.com/runopti/SpatialEvalLLM). The problems
describe a geometric construct composed of vertices and edges. At each vertex,
there is a unique object. An agent starts from one of the vertices, moves along
the edges and observes the objects at several vertices, and then after moving
for several steps along the edges, the job of the model is to determine what
object is at the final vertex where the agent stops.

We sampled from the hexagonal, circular, and rhombus constructs of
SpacialLLMEval. We also created similar constructs with tree structure,
triangular and diamond shapes and increased the difficulty compared to the
problems in SpacialLLMEval by increasing the number of hops of reasoning
(corresponding to the number of moves of the agent). Moreover, while the
original problems and the aforementioned problems we created mainly require
keeping track of the state after each move, we also create some variants of the
problem where we provide multiple paths that intersect at some vertex, thus
requiring backward reasoning from the intersection point to identify the
position of the previous objects. As an example, consider the problem below:

    You have been given a diamond tile map consisting of N rows [...] There is a
    unique object placed at each vertex. [...] You are initially at the top
    corner where you see a football. Then you move down-right for one step and
    see a shampoo. Then you move down-left for one step and you see a cat. [...]
    Then, you jump to a random vertex V where you see a bear. Then you move [...]
    Then you move up-left and you see a shampoo. Then you jump back to the
    random vertex V and do the following moves: down-left, down-left,
    down-right, up-left, down-left, up-left. What will you find?

For the first path (up until the first random jump), we know where the path
starts and we can use that along with the following moves to determine which
object is where. Then, a random jump is made to a vertex V but it is not
specified which vertex it is. However, we observe that after a number of moves,
the agent sees the **shampoo** again so it can reason backward from this point
to figure out which vertex it has been at in the previous steps. These
information can also be used to determine the vertex V which must then be used
to solve the problem when the second jump to V is made.
