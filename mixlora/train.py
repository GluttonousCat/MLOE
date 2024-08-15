from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM("mistralai/Mistral-7B-Instruct-v0.2",

)

print(model)


