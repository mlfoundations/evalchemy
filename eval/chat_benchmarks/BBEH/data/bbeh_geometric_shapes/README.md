# BBEH Geometric Shapes

SVG is a language for drawing shapes. We use two basic commands: 1- M (x, y)
corresponding to moving to the (x, y) coordinate, and 2- L (x, y) corresponding
to drawing a line from the current location to (x, y). We use the shape outlines
from [GeomVerse](https://arxiv.org/abs/2312.12241), a dataset of geometry
questions involving multiple shapes that share some elements, which are
specified as TikZ commands and convert them to SVG. We then ask the model
to identify what shapes will be drawn if we visualize the SVG.

We consider two extra axes for difficulty: 1- we randomly break some lines
segments into multiple collinear line segments, and 2- we add some extra lines
such that they intersect at some points and those intersections form some shapes
(in other cases, shapes are created using the full line segments and not at
their intersection points). We then create four subsets for the task
corresponding to the cross product of few vs many line breaks and intersect vs
no intersect.
