from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

_data_dir = './models'

if __name__ == '__main__': 
    # set HF_TOKEN in the .env for faster download time 
    load_dotenv()
    
    # beta 
    hf_hub_download(
        repo_id='unsloth/gemma-4-E2B-it-GGUF',
        filename='gemma-4-E2B-it-Q4_0.gguf',  
        local_dir=_data_dir
    )
    # alpha 
    hf_hub_download(
        repo_id='unsloth/gemma-4-E4B-it-GGUF',
        filename='gemma-4-E4B-it-Q4_0.gguf',  
        local_dir=_data_dir
    )

