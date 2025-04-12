# BBEH Buggy Tables

This task was constructed synthetically by the authors. The objective in this
task is to be able to respond to conditional queries over tabular data, where
the information in the table are presented in a buggy way but the description
for the bug is also presented so that the model can reconstruct the original
table based on that. As an example, we provide a row-major/column-major format
of the table where the null values have been mistakenly removed, but we also
provide the positions of the null values in the original table so one can
reconstruct the table given the two pieces of information. As another example,
we provide a buggy version of the table where some random values are appended at
the end of each row or each column, but we also specify how they have been added
so one can use this information to remove them and reconstruct the original
table. As yet another example, we provide a markdown format of the table that
mixes each two rows of the table into one row, but also provide an explanation
of how each two rows have been merged into one so that the original table can be
reconstructed based on that information. Examples of conditional queries
include computing some statistics (count, sum, mean, stdev, median) of some
columns while only considering rows where some columns have some specific
values.
