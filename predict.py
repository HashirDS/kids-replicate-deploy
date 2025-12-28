from cog import BasePredictor, Input, Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("🚀 Loading model...")
        
        base_model_id = "microsoft/phi-2"
        adapter_model_id = "hashir72/kids"

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

        # Load Base Model
        # device_map="auto" moves it to GPU automatically
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # Load Adapter (LoRA)
        # --- FIXED LINE BELOW: Added subfolder="Model" ---
        self.model = PeftModel.from_pretrained(
            model, 
            adapter_model_id, 
            subfolder="Model"
        )
        self.model.eval()

    def predict(
        self,
        prompt: str = Input(description="Input prompt for the model")
    ) -> str:
        """Run a single prediction on the model"""
        
        # Format input (Phi-2 often likes "Instruct: ... Output:" format, but we'll use your prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
