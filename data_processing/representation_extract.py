import torch
import os
import sys
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

if __name__ == '__main__':
    dataset_name = sys.argv[1]

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    model_name = '../models/DeepSeek-OCR'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True, torch_dtype=torch.float16)
    model = model.eval().cuda().to(torch.bfloat16)

    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    image_dir = f"../datasets/{dataset_name}/rendered_images/train"
    output_dir = f"../datasets/{dataset_name}/precomputed_visual_representations/train"
    os.makedirs(output_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(image_dir)):
        root_path = os.path.join(image_dir, img_name)
        
        origin_global_feature_list = []
        for page_id in range(len(os.listdir(root_path))):
            image_file = os.path.join(image_dir, img_name, f"page_{page_id:03d}.png")
            input_embeds, output_embeds, origin_global_features = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = './', base_size = 512, image_size = 512, crop_mode=False, save_results = True, test_compress = True)
            assert origin_global_features.shape == (8, 8, 1280)
            origin_global_feature_list.append(origin_global_features.unsqueeze(0))

        origin_global_feature = torch.cat(origin_global_feature_list, dim=0)   
        assert origin_global_feature.shape == (len(os.listdir(root_path)), 8, 8, 1280) 
        output_file = os.path.join(output_dir, img_name + ".pt")
        torch.save(origin_global_feature.cpu(), output_file)
    