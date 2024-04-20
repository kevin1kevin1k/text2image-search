from transformers import AutoProcessor, AutoTokenizer, CLIPModel

pretrained_model_name = "openai/clip-vit-base-patch32"
processor = AutoProcessor.from_pretrained(pretrained_model_name)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
model = CLIPModel.from_pretrained(pretrained_model_name)

PROJECTION_DIM = model.config.projection_dim
