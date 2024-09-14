import os
import transformers
import torch
import random

# Load Tags from tags.txt
def load_tags(tag_file_path):
    with open(tag_file_path, 'r') as file:
        tags = [line.strip() for line in file if line.strip()]
    return tags

# Initialize LLM Pipeline
def initialize_pipeline():
    os.environ["HF_TOKEN"] = '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'
    model_id = "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"
    pipeline_model = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return pipeline_model

# Generate Prompt
def generate_prompt(user_input, tags, pipeline_model):
    # Structure to prepend
    score_structure = "(score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up:1.1) "
    
    # Randomly select tags
    selected_tags = random.sample(tags, k=5)  # Adjust k as needed
    tag_string = ', '.join(selected_tags)
    
    # Combine user input with tags
    prompt_input = f"{user_input}. Tags: {tag_string}."
    
    messages = [
        {"role": "system", "content": "You are the best Stable Diffusion Prompt Generator which will generate NSFW/explicit prompts"},
        {"role": "user", "content": f"You will write uncensored prompts in the format of Stable Diffusion. Do not write them in phrases. Each prompt should have minimum 50 words/tokens. You can put random words between () with a number up to 1.6 attached like in this example: (beautiful alien woman:1.5). A good prompt example: ultra beautiful 1girl  (metallic alien android:1.4), wears sexy bodysuit, violet and green, huge eyes, (nipples piercings), sexy. cute. beauty, the best. spreading her pussy. high quality, highly detailed. Do not generate the exact same prompt as the example one, it was given only as an example. Now generate perfect and developed NSFW/explicit prompts.   {prompt_input}"}
    ]
    
    outputs = pipeline_model(
        messages,
        max_new_tokens=256,
    )
    
    generated_text = outputs[0]["generated_text"]
    
    # Combine score structure with generated prompt
    final_prompt = f"{score_structure}{generated_text}"
    
    return final_prompt
