# Facencial Customer Service AI — SFT & Distillation

## 📌 Project Description

This project adapts the **DeepSeek-R1-0528-Qwen3-8B** model into a **Facencial Customer Service AI** with reasoning capabilities, leveraging **Supervised Fine-Tuning (SFT)** and **knowledge distillation** from a teacher model (**GPT-5**). The training uses a synthetic dataset designed to answer customer queries accurately via retrieval-augmented knowledge.

**Dataset Source:** [Facencial-CoT-Updated Dataset on Hugging Face](https://huggingface.co/datasets/MMumtazSakho/facencial-cot-updated)
**Fine-Tuned Model:** [Facencial-CoT-LoRA-Updated-v2 on Hugging Face](https://huggingface.co/MMumtazSakho/Facencial-CoT-Lora-updated-v2)

---

## 🚀 Key Features

* **Reasoning Chat Template** with `<think>` and `<output>` tags to separate thought process from the final answer.
* **LoRA Fine-Tuning** with configurable rank.
* **Automated Evaluation & Correction Dataset** creation using GPT-5 as evaluator.
* **Distillation Pipeline** to compress the model while retaining reasoning ability.
* **Correction Dataset Storage** for iterative training.

---

## 📈 Accuracy Progression

| Phase     | Accuracy |
| --------- | -------- |
| Phase I   | 36%      |
| Phase II  | 68%      |
| Phase III | 86%      |

> Accuracy improves as more expert-labeled RAG data from GPT-5 is integrated.
> All of the testing case available in Bajau_submission_updated.ipynb

---

## 📂 Notebook Structure

1. **Installation** – Install dependencies like `unsloth`, `vllm`, `bitsandbytes`, `peft`, and `trl`.
2. **Model Loading** – Load DeepSeek model via `FastLanguageModel` with LoRA rank, sequence length, and GPU memory optimization.
3. **Prompt Template** – Apply custom reasoning/answer separation template.
4. **Dataset Preparation** – Format into `problem`, `generated_solution`, and `expected_answer`.
5. **Evaluation Pipeline** – GPT-5 evaluates and generates corrections.
6. **Distillation** – Transfer teacher knowledge to student model.
7. **Export Dataset** – Save datasets in `.pkl` or `.json`.

---

## ⚙️ How to Run

### 1. Clone Repository

```bash
git clone <repo_url>
cd <repo_folder>
```

### 2. Install Dependencies (Colab Example)

```bash
!pip install --no-deps unsloth vllm==0.8.5.post1 bitsandbytes accelerate xformers==0.0.29.post3 peft trl cut_cross_entropy unsloth_zoo
!pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
```

### 3. Run the Notebook

* Upload dataset.
* Configure LoRA rank and hyperparameters.
* Run all cells sequentially.

---

## 📊 Dataset Example

```json
{
  "problem": "What is the price of Package D?",
  "generated_solution": "Let's think step by step...<output>Package D costs Rp50,000 for 100 credits (Rp500/photo).</output>",
  "expected_answer": "Package D costs Rp50,000 for 100 credits (Rp500/photo)."
}
```


## LLM Fine-Tuning for Task-Oriented Instruction Generation

### 1. Model Choice

**DeepSeek-R1-0528-Qwen3-8B** was chosen for its:

* Strong reasoning and instruction-following.
* Open-source licensing.
* LoRA/QLoRA compatibility.
* Efficient inference on quantized GPUs.

### 2. Dataset Design

* **Fields**:

  * `problem`: user intent.
  * `generated_solution`: chain-of-thought reasoning.
  * `expected_answer`: structured, concise instructions.
* **Collection**:

  1. Extract queries from docs, support tickets.
  2. Annotate manually or with GPT-5.
  3. Ensure domain diversity.
* **Preprocessing**:

  * Format into `<think>` and `<output>`.
  * Tokenize and check context length.
* **Edge Cases**:

  * Scrub sensitive info.
  * Maintain domain/topic diversity.

### 3. Fine-Tuning Strategy

* **LoRA-based instruction tuning**.
* **Hyperparameters**: LR=1e-5\~5e-5, LoRA rank & alpha, batch size, 5 epochs.
  
### 4. Evaluation & Benchmarking

* **Qualitative**:

  * Human review.
  * GPT-5 + RAG evaluation.
* **Benchmarking**:

  * Compare with base and human gold.
  * Use fixed multi-domain test set.
* **Assessment**:

  * Automatic NLP metrics.

---

## 🔮 Future Improvements

* **GRPO**: RL-based output alignment beyond supervised fine-tuning.
* **DPO**: Optimize with AI/human preference pairs.
* **Self-Reflection**: Model critiques and revises its own answers.
* **Dynamic Knowledge Updates**: Keep retrieval base in sync with new Facencial info.
* **Multi-Metric Evaluation**: Include politeness, conciseness, and clarity scoring.

