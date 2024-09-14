import gradio as gr
from utils.prompt_generator import load_tags, initialize_pipeline, generate_prompt

# Initialize Pipeline and Load Tags
pipeline_model = initialize_pipeline()
tags = load_tags('/content/promptgen/app/utils/tags.txt')

# Define Gradio Interface Function
def prompt_generator(user_input):
    prompt = generate_prompt(user_input, tags, pipeline_model)
    return prompt

# Create Gradio Interface
iface = gr.Interface(
    fn=prompt_generator,
    inputs=gr.Textbox(lines=2, placeholder="Enter a theme or keyword for your Stable Diffusion prompt..."),
    outputs=gr.Textbox(label="Generated Stable Diffusion Prompt"),
    title="Stable Diffusion Prompt Generator",
    description="""
    Enter a theme or keyword, and the generator will create a detailed Stable Diffusion prompt for you.
    Each prompt starts with a predefined score structure for better customization.
    """,
    examples=[
        ["Serene landscape with mountains and rivers"],
        ["Cyberpunk city at night"],
        ["Fantasy dragon flying over a castle"]
    ],
)

# Launch the Interface with Share Link
iface.launch(share=True)
