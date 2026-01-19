import torch
from transformers import pipeline

model_id = "llava-hf/llama3-llava-next-8b-hf"

def create_pipeline_with_fallback(model_id):
	# Try a sequence of dtypes to avoid mat1/mat2 dtype mismatches
	for dt in (torch.float16, torch.bfloat16, torch.float32):
		try:
			# device_map='auto' helps place weights consistently; torch_dtype sets model dtype
			return pipeline("image-text-to-text", model=model_id, torch_dtype=dt, device_map="auto")
		except Exception:
			# try next dtype
			continue
	# final fallback: force CPU with float32
	return pipeline("image-text-to-text", model=model_id, torch_dtype=torch.float32, device="cpu")

pipe = create_pipeline_with_fallback(model_id)

messages = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "You have a \"get_weather\" tool, when called without parameters, you will get the weather. Use it to answer the question: Do I need an umbrella today?"},
        ],
    },
]

out = pipe(text=messages, max_new_tokens=20)
print(out)