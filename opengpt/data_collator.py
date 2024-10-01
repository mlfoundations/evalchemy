import torch

class DataCollatorWithPadding(object):
    r''' Will pad or trim examples to the appropriate length.
    '''
    def __init__(self, pad_token_id, ignore_index, max_seq_len):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.max_seq_len = max_seq_len

    def __call__(self, instances):
        input_ids, labels = tuple([torch.tensor(instance[key][0:self.max_seq_len]) for instance in instances] for key in ("input_ids", "labels"))
        batch = {}
        
        batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id) 
        batch['labels'] = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.ignore_index)
        batch['attention_mask'] = batch['input_ids'].ne(self.pad_token_id)
    
        return batch