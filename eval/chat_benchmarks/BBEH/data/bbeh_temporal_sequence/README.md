# BBEH Temporal Sequence

In this task, the calendar schedules of a few people is provided for an entire
week. The blocked times for the calendar of each person is sampled randomly, and
is provided as text either by giving the times of the day when it is blocked or
giving the times of the day when it is free. The goal is to find: 1- the longest
meeting that can be scheduled for them, and 2- the number of possibilities for
such a meeting. These people may also have some constraints or we might have
some information about them that has to be taken into account for meeting
scheduling. Examples include: being in a different timezone than the other
participants, needing some free time before/after the meeting, being flexible to
miss a portion of the meeting, requiring some free time for lunch, only being
able to attend meetings of up to a certain length, being willing to free up some
specific parts of the day if needed, etc.
