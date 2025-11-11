# predict.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

class Predictor:
    def __init__(self):
        print("🚀 Loading model...")
        base_model = "microsoft/phi-2"         # Base model
        adapter_model = "hashir72/kids"        # Your fine-tuned adapter on Hugging Face

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Load base + adapter (LoRA)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model = PeftModel.from_pretrained(model, adapter_model)
        self.model.eval()

    def predict(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
