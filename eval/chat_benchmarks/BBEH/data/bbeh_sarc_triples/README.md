<!-- mdlint off(LINE_OVER_80) -->

# BBEH SARC Triples

[SARC](https://aclanthology.org/L18-1102.pdf) (Self-Annotated Corpus for Sarcasm) is a large dataset of sarcasm responses mined from the Reddit social media / forum platform. Many Reddit users end a post or reply with the token **/s** when they have intended the preceding text to be interpreted sarcastically or satirically. This allowed positive examples of user-intended sarcasm to be mined.

Forking off the SARC dataset, we construct a challenging task for LLMs that requires reading three independent examples from SARC, and classifying each into binary label, where a positive label indicates sarcasm. The SARC authors created a balanced test set with 64,666 examples. Many of these examples can only be understood with an image or an article link that accompanied the original post or reply. On the other hand, some examples, usually with longer textual content, can be understood on their own. We design our derived benchmark to consist mainly of the latter type. To achieve this, we filter out examples with either (1) less than 100 characters or (2) without a reply, resulting in 679 examples from the original test set, with 48.4% positive label rate. We sample (uniformly-at-random) 600 examples from this set, group them (uniformly-at-random) into groups of three, and pass the text of each 3-tuple of post, reply pair to the following prompt:

    Here are three (post, reply) pairs from Reddit. Your task is to decide whether each reply is sarcastic. Specifically, label each pair with a "0" or "1", where a "1" indicates that the reply is sarcastic, and a "0" indicates that the reply does not contain sarcasm, and provide your final answer as a comma-separated set of labels (e.g., "1,0,0" or "0,0,0").
    POST 1: post1_text
    REPLY 1: reply1_text
    POST 2: post2_text
    REPLY 2: reply2_text
    POST 3: post3_text
    REPLY 3: reply3_text
