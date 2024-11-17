# Code for generating data in MetaMath

> Tips
- set `export APIKEY="YOUR_API_KEY"` in `~/.bash_profile` 
- argument `--num_repeat`: #outputs from ChatGPT for each input by tempearture sampling
- use argument `--part` to void overwriting previous generated data
- use argument `--cont` to continue a previous generating procedure, otherwise, will re-fetch data

### 0. Create backward questions
```bash
cd code
bash -x run_create_backward_questions.sh
```

### 1. AnsAug

```bash
cd code
bash -x run_forward.sh
```

### 2. Rephrasing

```bash
cd code
bash -x run_rephrase.sh
```

### 3. Self-Verification

```bash
cd code
bash -x run_sv.sh
```

### 4. FOBAR

```bash
cd code
bash -x run_backward.sh
```