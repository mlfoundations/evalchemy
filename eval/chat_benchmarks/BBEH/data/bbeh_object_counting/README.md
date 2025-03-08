# BBEH Object Counting

Given a long list of objects that a person has, the model has to count the
number of items of a certain type. For examples, the items might belong to
classes (fruits, cell phones, cars) and the goal may be to count the total
number of cell phones that the person has. We consider two types of questions:
1- counting the sum of the number of items belonging to two different classes,
and 2- finding the absolute difference of the number of items belonging to two
different classes. To add to the difficulty of the task, some irrelevant
information, including the number of the same items that other people have,
are added to the input context so the problem becomes one of finding multiple
needles in a haystack.
