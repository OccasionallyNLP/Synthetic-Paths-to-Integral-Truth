import torch
from accelerate import Accelerator
from accelerate import PartialState
from accelerate.utils import gather_object
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, GenerationConfig, TextStreamer, BitsAndBytesConfig
from utils.utils import *
import pickle
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_beams", type=int)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--do_sample", type=str2bool, default=False)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--get_train_prompt',type=str2bool, default=True)

    args = parser.parse_args()
    return args


def get_tokenizer_and_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map={"": accelerator.process_index}, attn_implementation="flash_attention_2", trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer.padding_side='left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model#, generation_config

# previous version (~24.07.21)
def get_prompt(tokenizer, document, question, tokenize=False):
    user_message = f"""
    Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).

{document}

Question: {question}
Answer:
    """
    messages = [
            {"role": "user", "content": user_message}
            ]
    prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=tokenize, 
            add_generation_prompt=True
            )
    return prompt

def get_qa_prompt(tokenizer, document, question, tokenize=False):
    user_message = f"""Answer the question based on the given document.
    Document : {document}
    Question: {question}
    Answer:"""
    messages = [
            {"role": "user", "content": user_message},
            ]
    prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=tokenize, 
            add_generation_prompt=True
            )
    return prompt



if __name__ == "__main__":
    seed_everything(42)
    args = get_args()
    os.makedirs(args.output_dir, exist_ok = True)
    print(args)
    accelerator = Accelerator()
    accelerator.wait_for_everyone()    
#     start=time.time()
    ###########################################################################################
    # tokenizer, config, model
    ###########################################################################################
    ##################################################################################
    tokenizer, model  = get_tokenizer_and_model(args)
    
    test_data = load_jsonl(args.test_data)#'/home/work/user/ocw/curriculum_dpo/data/evaluation/halubench/halubench.jsonl')
    batch_size = args.batch_size
    
    # We set it to 8 since it is better for some hardware. More information here 
    pad_to_multiple_of = 8
    padding_side_default = tokenizer.padding_side
    tokenizer.padding_side = "left"
    for i in test_data:
        if args.get_train_prompt==False:
            prompt = get_prompt(tokenizer, '\n\n'.join(i['segments']).strip(), i['question'], tokenize=False)
        else:
            prompt = get_qa_prompt(tokenizer, '\n\n'.join(i['segments']).strip(), i['question'], tokenize=False)
        i['input'] = prompt
    formatted_prompts = [[j['input'] for j in test_data[i : i + batch_size]] for i in range(0, len(test_data), batch_size)]
    tokenized_prompts = [
    tokenizer(formatted_prompt, padding=True, pad_to_multiple_of=pad_to_multiple_of, add_special_tokens=False, return_tensors="pt")
    for formatted_prompt in formatted_prompts
    ]
    tokenizer.padding_side = padding_side_default
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.encode('\n',add_special_tokens=False)[0]
        ]
    streamer = None
    if accelerator.is_main_process and batch_size == 1:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) 
    completions_per_process = []
    with accelerator.split_between_processes(tokenized_prompts, apply_padding=True) as batched_prompts:
        for batch in tqdm(batched_prompts,disable=accelerator.is_main_process != True):
            batch = batch.to(accelerator.device)
            length = batch['input_ids'].size(1)
            print(length)
            # just greedy
            outputs = model.generate(**batch, streamer=streamer, eos_token_id=terminators, pad_token_id = tokenizer.pad_token_id, num_beams=1, max_new_tokens=args.max_new_tokens, temperature=1.0, do_sample=args.do_sample)
            outputs = outputs[:,length:].contiguous()
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            accelerator.wait_for_everyone()   
            completions_per_process.extend(generated_text)
            
    completions_gather = gather_object(completions_per_process)
    
    completions = completions_gather[: len(test_data)]
    accelerator.wait_for_everyone()   
    if accelerator.is_main_process:
        ####################################################################################################################################
        for i,j in zip(test_data, completions):
            i['predict']=j
        ###########################################################################################################################################
        save_jsonl(args.output_dir, test_data, 'attached')
        with open(os.path.join(args.output_dir,'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
