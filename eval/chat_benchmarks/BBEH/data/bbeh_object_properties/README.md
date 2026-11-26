# BBEH Object Properties

In this task, an initial collection of objects with different properties (color,
size, origin, smell, and material) are provided (e.g., a extra-small blue
Canadian jar made of glass and with a smell of rose). Then, the collection goes
through several updates corresponding to adding, removing or editing some of the
objects. The updates are explained in the prompt and the models require a full
grasp of the object properties to identify what changes to the collection must
be made for each update. A simple example of an update is as follows:

    My dad threw away all objects of a certain color from my collection.
    After this, my collection only had 5 blue objects and 3 white objects.

For the above update, one has to find which color has been removed by comparing
the new colors with the object colors in the previous collection, and then
update the collection accordingly. The set of updates that the collection goes
through in each of the examples are randomly selected from a large set of
possible changes. At the end, a question is asked about the final collection.
The question is either an **either** question in which we ask how many items in
the final collection have property 1 or property 2, ... (e.g., how many items
are either blue or small), or a **neither** question in which we ask how many
items neither have property 1 nor property 2, ... (e.g., how many items are not
blue and are not small).
