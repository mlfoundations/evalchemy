name: opengpt
operators:
  - id: load_alpaca
    config:
      type: hf_source
      dataset: yahma/alpaca-cleaned
      split: train
      columns: 
        - instruction
        - output
      num_truncate: 5

  - id: link_generation
    config:
      type: function
      function: data_strategies.OpenGPT.utils.get_health_az_links
    input_ids:
      - load_alpaca
    
  - id: create_health_az_table
    config:
      type: function
      sharded: True
      function: data_strategies.OpenGPT.utils.create_health_az_table
    input_ids:
      - link_generation

  - id: add_annotations
    config:
      type: function
      sharded: True
      function: data_strategies.OpenGPT.utils.create_health_az_table
      function_config:
        config_path: external_repositories/OpenGPT/OpenGPT/configs/openhermes_regneration.yaml
    input_ids:
      - create_health_az_table

  - id: parse_outputs
    config:
      type: function
      sharded: True
      function: data_strategies.OpenGPT.utils.parse_data
      function_config:
        instruction_column: "prompt"
        completion_column: "completion"
    input_ids:
      - parse_outputs