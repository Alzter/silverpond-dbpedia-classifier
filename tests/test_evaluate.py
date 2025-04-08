import pytest
import evaluate as ev
import finetune as ft
import model_prompts
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import torch

@pytest.fixture
def llm():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4", # QLoRA uses 4-bit NormalFloat precision,
        bnb_4bit_use_double_quant = True, # QLoRA uses double quantising,
        bnb_4bit_compute_dtype = torch.float16
    )

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return (model, tokenizer)

@pytest.fixture
def eval_dataset():
    eval_data = load_dataset("fancyzhx/dbpedia_14", split="test")
    eval_data = ft.sample_dataset(eval_data, labels_column="label", samples_per_class=2)
    eval_data, label_names = ft.preprocess_dataset(eval_data, text_column="content", labels_column="label")
    return (eval_data, label_names)

def test_evaluate_llm(llm, eval_dataset):
    model, tokenizer = llm
    eval_data, label_names = eval_dataset

    eval_config = ev.EvaluationConfig(
        name="Zero-shot",
        prompt = model_prompts.PROMPT_ZEROSHOT,
        max_tokens = 3
    )

    result = ev.evaluate(
        model, tokenizer,
        label_names=label_names,
        eval_dataset=eval_data,
        eval_config=eval_config
    )



