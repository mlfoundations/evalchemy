# LLM Benchmark Reproduction Results

| Benchmark   | Tester  | Model                                   | Metric/Score                  | Our Results | Reported Results | Reported Results Source             |
|-------------|---------|-----------------------------------------|-------------------------------|-------------|------------------|-------------------------------------|
| Alpaca Eval | Etash   | Meta LLama 3 8b Instruct                | win_rate                      | 24.04       | 22.57            |                                     |
|             |         |                                         | standard_error                | 1.27        | 1.26             |                                     |
|             |         |                                         | avg_length                    | 1937        | 1899             |                                     |
|             |         |                                         | length_controlled_winrate     | 24.24       | 22.92            |                                     |
| HumanEval   | Etash   | DeepSeek Coder 6.7B Instruct            | Python                        | 0.798       | 78.9%            |                                     |
|             |         |                                         | Bash                          | 0.354       | 36.7%            |                                     |
| MBPP        | Etash   | DeepSeek Coder 6.7B Instruct            | pass@1                        | 0.64        | 65.4%            |                                     |
| RepoBench   | Negin   | StarCoder                               | EM (cross_file_first)         | 28.0        | 28.0             |                                     |
|             |         |                                         | ES (cross_file_first)         | 67.37       | 69.6             |                                     |
|             |         |                                         | EM (cross_file_random)        | 37.08       | 37.3             |                                     |
|             |         |                                         | ES (cross_file_random)        | 71.28       | 73.69            |                                     |
|             |         |                                         | EM (in_file)                  | 34.17       | 33.8             |                                     |
|             |         |                                         | ES (in_file)                  | 70.46       | 72.37            |                                     |
|             |         |                                         | EM (weighted avg)             | 31.69       | 31.69            |                                     |
|             |         |                                         | ES (weighted avg)             | 69.09       | 71.2             |                                     |
|             |         | Codegen-350M-mono                       | EM (cross_file_first)         | 15.27       | 15.14            |                                     |
|             |         |                                         | ES (cross_file_first)         | 58.03       | 60.1             |                                     |
|             |         |                                         | EM (cross_file_random)        | 27.7        | 27.7             |                                     |
|             |         |                                         | ES (cross_file_random)        | 67.33       | 68.9             |                                     |
|             |         |                                         | EM (in_file)                  | 25.11       | 25.2             |                                     |
|             |         |                                         | ES (in_file)                  | 66.28       | 67.8             |                                     |
|             |         |                                         | EM (weighted avg)             | 22.12       | 20.7             |                                     |
|             |         |                                         | ES (weighted avg)             | 62.9        | 64.2             |                                     |
|             |         | Codegen-2B-mono                         | EM (cross_file_first)         | 22.12       | 22.1             |                                     |
|             |         |                                         | ES (cross_file_first)         | 62.9        | 64.9             |                                     |
|             |         |                                         | EM (cross_file_random)        | 34.18       | 34.4             |                                     |
|             |         |                                         | ES (cross_file_random)        | 71.12       | 72.6             |                                     |
|             |         |                                         | EM (in_file)                  | 31.14       | 31.2             |                                     |
|             |         |                                         | ES (in_file)                  | 69.61       | 70.93            |                                     |
|             |         |                                         | EM (weighted avg)             | 27.26       | 27.3             |                                     |
|             |         |                                         | ES (weighted avg)             | 66.57       | 68.3             |                                     |
|             |         | Codegen-6B-mono                         | EM (cross_file_first)         | 26.81       | 26.9             |                                     |
|             |         |                                         | ES (cross_file_first)         | 66.23       | 67.9             |                                     |
|             |         |                                         | EM (cross_file_random)        | 38.28       | 38.3             |                                     |
|             |         |                                         | ES (cross_file_random)        | 73.34       | 74.5             |                                     |
|             |         |                                         | EM (in_file)                  | 34.9        | 34.96            |                                     |
|             |         |                                         | ES (in_file)                  | 71.21       | 72.59            |                                     |
|             |         |                                         | EM (weighted avg)             | 31.56       | 31.67            |                                     |
|             |         |                                         | ES (weighted avg)             | 69.16       | 70.68            |                                     |
| MTBench     | Etash   | stabilityai/stablelm-tuned-alpha-7b     |                               | 1.0         | 2.75             |                                     |
|             |         | Nexusflow/Starling-LM-7B-beta           |                               | 7.69        | 8.12             |                                     |
|             |         | mistralai/Mistral-7B-Instruct-v0.1      |                               | 6.65        | 6.84             |                                     |
|             |         | databricks/dolly-v2-12b                 |                               | 1.087       | 3.28             |                                     |
| WildBench   | Etash   | princeton-nlp/gemma-2-9b-it-SimPO       |                               | 5.083       | 5.33             |                                     |
|             |         | meta-llama/Meta-Llama-3-8B-Instruct     |                               | 3.27        | 2.92             |                                     |
| IFEval      | Negin   | meta-llama/Llama-3.1-8B-Instruct        |                               | 79.1        | 80.4             |                                     |
| ZeroEval    | Negin   | meta-llama/Llama-3.1-8B-Instruct        | crux                          | 40.75       | 39.88            |                                     |
|             |         |                                         | math-l5                       | 24.69       | 22.19            |                                     |
|             |         |                                         | zebra                         | 11.70       | 12.8             |                                     |
| MixEval     | Negin   | Meta-Llama-3-8B-Instruct                | overall                       | 73.0        | 75.0             |                                     |
|             |         |                                         | TriviaQA                      | 67.5        | 71.7             |                                     |
|             |         |                                         | MMLU                          | 71.3        | 71.9             |                                     |
|             |         |                                         | HellaSwag                     | 66.3        | 65.7             |                                     |
| AMC23       | Ryan    | Qwen/Qwen2.5-Math-1.5B-Instruct         | accuracy                      | 24/40       | 24/40            | [Qwen2.5-Math Paper](https://arxiv.org/abs/2409.12122v1) Table 5 |
| AIME24      | Ryan    | Qwen/Qwen2.5-Math-1.5B-Instruct         | accuracy                      | 3/30        | 3/30             | [Qwen2.5-Math Paper](https://arxiv.org/abs/2409.12122v1) Table 5 |
|             |         | Qwen/Qwen2.5-32B-Instruct               | accuracy                      | 6/30        | 5/30             | [Sky-T1 Blog Post](https://novasky-ai.github.io/posts/sky-t1/) |
|             |         | Qwen/QwQ-32B-Preview                    | accuracy                      | 13/30       | 15/30            | [Sky-T1 Blog Post](https://novasky-ai.github.io/posts/sky-t1/) |
|             |         | NovaSky-AI/Sky-T1-32B-Preview           | accuracy                      | 13/30       | 13/30            | [Sky-T1 Blog Post](https://novasky-ai.github.io/posts/sky-t1/) |
| MATH500     | Ryan    | Qwen/Qwen2.5-32B-Instruct               | accuracy                      | 78.6        | 76.2             | [Sky-T1 Blog Post](https://novasky-ai.github.io/posts/sky-t1/) |
|             |         | NovaSky-AI/Sky-T1-32B-Preview           | accuracy                      | 84.0        | 82.4             | [Sky-T1 Blog Post](https://novasky-ai.github.io/posts/sky-t1/) |
|             |         | Qwen/QwQ-32B-Preview                    | accuracy                      | 83.6        | 85.4             | [Sky-T1 Blog Post](https://novasky-ai.github.io/posts/sky-t1/) |
| BigCodeBench| Marianna| Qwen/Qwen2.5-14B-Instruct               | instruct (pass@1)             | 41.5        | 39.8             |                                     |
|             |         |                                         | complete (pass@1)             | 52.6        | 52.2             |                                     |
|             |         | meta-llama/Meta-Llama-3.1-8B-Instruct   | instruct (pass@1)             | 30.7        | 32.8             |                                     |
|             |         |                                         | complete (pass@1)             | 41.9        | 40.5             |                                     |
|             |         | Qwen/Qwen2.5-7B-Instruct                | instruct (pass@1)             | 35.2        | 37.6             |                                     |
|             |         |                                         | complete (pass@1)             | 46.7        | 46.1             |                                     |
| LiveBench   | Negin   | Meta-Llama-3-8B-Instruct                | global_average                | 26.78       | 26.74            | https://livebench.ai/#/?q=3.1
|HumanEvalPlus| Sedrick | mistralai/Mistral-7B-Instruct-v0.2      | accuracy (pass@1)             | 27.44       | 36.0             | [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html) |
|             |         | meta-llama/Llama-3.1-8B-Instruct        | accuracy (pass@1)             | 62.2        | 62.8             | [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html) |
|             |         | google/codegemma-7b-it                  | accuracy (pass@1)             | 36.6        | 51.8             | [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html) |
| MBPPPlus    | Sedrick | mistralai/Mistral-7B-Instruct-v0.2      | accuracy (pass@1)             | 43.9        | 37.0             | [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html) |
|             |         | meta-llama/Llama-3.1-8B-Instruct        | accuracy (pass@1)             | 58.7        | 55.6             | [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html) |
|             |         | google/codegemma-7b-it                  | accuracy (pass@1)             | 56.6        | 56.9             | [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html) |
|LiveCodeBench| Negin   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | (pass@1)                      | 37.9        | 37.6             | [DeepSeek-R1 Paper](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf) Table 5 |
| GPQADiamond | Negin   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | accuracy                      | 52          | 49.1             | [DeepSeek-R1 Paper](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf) Table 5 |
| MultiPL-E.  | Marianna| Qwen/CodeQwen1.5-7B-Chat                | java (pass@1)                 | 64.0        | 61.04            |  [Big Code Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)                                     |
|             |         |                                         | cpp (pass@1)                  | 67.85       | 61.0             |   [Big Code Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)                                    |
|             |         |                                         | cs (pass@1)                   | 52.6        | 52.2             |    [Big Code Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)                                   |
|             |         |                                         | php (pass@1)                  | 64.7        | 69.22            |   [Big Code Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)                                    |
|             |         |                                         | sh (pass@1)                   | 37.0        | 39.2             |    [Qwen2.5-Coder Technical Report](https://arxiv.org/pdf/2409.12186) (Table 17)
|             |         |                                         | ts (pass@1)                   | 73.0        | 71.7             |           [Qwen2.5-Coder Technical Report](https://arxiv.org/pdf/2409.12186) (Table 17)                          |
|             |         |                                         | js (pass@1)                   | 69.5        | 75.2            |           [Qwen2.5-Coder Technical Report](https://arxiv.org/pdf/2409.12186) (Table 17)                          |                                 |
| CRUXEval    | Marianna| Qwen/Qwen2.5-Coder-7B-Instruct          | Input-CoT (pass@1)            | 66.2        | 65.8            |  [Qwen2.5-Coder Technical Report](https://arxiv.org/pdf/2409.12186) (Table 18)                                     |
|             |         |                                         | Output-CoT (pass@1)           | 66.7        | 65.9            |                                     |
|             |         | Qwen/Qwen2.5-Coder-3B-Instruct          | Input-CoT (pass@1)            | 53.4        | 53.2            |                                    |
|             |         |                                         | Output-CoT (pass@1)           | 53.3        | 56.0            |                                     |
| MedQA       | Asad    | openai/gpt-4o-mini-2024-07-18           | accuracy                      | 74.4%       | 72.4%           | [Vals.ai Leaderboard](https://www.vals.ai/benchmarks/medqa-02-03-2025) |