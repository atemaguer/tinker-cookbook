import tinker
from tinker import types
# Create some training examples
# service_client = tinker.ServiceClient(base_url="http://44.195.1.40:8000", api_key="dummy")
service_client = tinker.ServiceClient()


# print("Available models:")
# for item in service_client.get_server_capabilities().supported_models:
#     print("- " + item.model_name)

# training_client = service_client.create_lora_training_client(
#     base_model="meta-llama/Llama-3.2-1B",
# )


training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    rank=1,
)

# Get the tokenizer from the training client
tokenizer = training_client.get_tokenizer()
 
sampling_client = training_client.save_weights_and_get_sampling_client(name='test_sampler')
 
# Now, we can sample from the model.
prompt=types.ModelInput.from_ints(tokenizer.encode("What is the capital of Uganda?"))
params = types.SamplingParams(temperature=0.0, stop=["<|endoftext|>", "<|im_end|>"]) # Greedy sampling
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
result = future.result()

print(tokenizer.decode(result.sequences[0].tokens))
