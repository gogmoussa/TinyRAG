from llama_cpp import Llama

llm = Llama(
    model_path="local_model.gguf", 
    n_ctx=2048,
    n_threads=4  # adjust for your CPU
)

response = llm("Q: What gear should I bring for backcountry camping?\nA:", max_tokens=128)
print(response["choices"][0]["text"])