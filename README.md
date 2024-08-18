# Few-shot evaluation

This project, which is based on repository `EleutherAI/lm-evaluation-harness`, focuses on evaluating the performance of Large Language Model (LLM) model with few-shot learning on some certain tasks (such as multiple-choice, sentence completement,... benchmarks). We implement 4 benchmarks: vi_lambada, vi_wikipediaqa, vi_comprehension, and vi_exams (with 7 subjects) from ViLLM paper, in order to test the performance of LLMs for Vietnamese dataset. Details will be included in the Reference links section below.

You could use your Google Colab, Kaggle, or even your own computer (with GPU for faster computation) in order to run this project!

1. **Clone the github repository:** `git clone https://github.com/luongdinhdung123/lm-evaluation-harness`. 

2. **Install the required libraries:**
    - Open the folder: `cd lm-evaluation-harness`
    - Install the required libraries: `pip install -e .`

3. **Run the evaluation for your models:**

- Here is how to run the evaluation of a LLM model for vi_wikipediaqa benchmark on Kaggle (You should check your `output_path` if you run on other platforms):
```bash
lm_eval \
--model hf \
--model_args pretrained=pretrained=vinai/PhoGPT-4B,trust_remote_code=True \
--tasks wikipediaqa_vi \
--num_fewshot 5 \
--device cuda:0 \
--output_path /kaggle/working/lm-evaluation-harness \
--log_samples \
```

- For vi_lambada:

```bash
lm_eval \
--model hf \
--model_args pretrained=pretrained=vinai/PhoGPT-4B,trust_remote_code=True \
--tasks lambada_vi \
--num_fewshot 5 \
--device cuda:0 \
--output_path /kaggle/working/lm-evaluation-harness \
--log_samples \
```

- For vi_comprehension:
```bash
lm_eval \
--model hf \
--model_args pretrained=pretrained=vinai/PhoGPT-4B,trust_remote_code=True \
--tasks comprehension_vi \
--num_fewshot 5 \
--device cuda:0 \
--output_path /kaggle/working/lm-evaluation-harness \
--log_samples \
```

- For vi_exams:
```bash
lm_eval \
--model hf \
--model_args pretrained=pretrained=vinai/PhoGPT-4B,trust_remote_code=True \
--tasks exams_diali_vi, exams_lichsu_vi, exams_hoahoc_vi, exams_sinhhoc_vi, exams_vatli_vi, exams_toanhoc_vi, exams_vanhoc_vi \
--num_fewshot 5 \
--device cuda:0 \
--output_path /kaggle/working/lm-evaluation-harness \
--log_samples \
```

- You can read the `docs` and `examples` folder to understand more!
4. **Reference Links:**
    - [Github Repo for References](https://github.com/EleutherAI/lm-evaluation-harness)
    - [Paper: ViLLM-Eval: A Comprehensive Evaluation Suite for
Vietnamese Large Language Models](https://arxiv.org/pdf/2404.11086)