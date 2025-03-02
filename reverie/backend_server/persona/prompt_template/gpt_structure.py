"""
Author: Joon Sung Park (joonspk@stanford.edu)
Modified by Chaehyeon Kim

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import json
import random
# import openai
# Replacing with Llama model
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
import torch
import time 

from utils import *

# openai.api_key = openai_api_key

# Llama model specifications
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Adjust model size as needed
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically assigns model parts to available GPUs/CPUs
    torch_dtype=torch.float16  # Use float16 for efficiency (if hardware supports it)
)

def temp_sleep(seconds=0.1):
  time.sleep(seconds)

def ChatGPT_single_request(prompt): 
  temp_sleep()

  # completion = openai.ChatCompletion.create(
  #   model="gpt-3.5-turbo", 
  #   messages=[{"role": "user", "content": prompt}]
  # )
  
  # return completion["choices"][0]["message"]["content"]

  # Below, code is modified to run on llama
  # Formatting as user input
  formatted_prompt = f"[INST] {prompt} [/INST]"
  input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
  with torch.no_grad():
    output = model.generate(input_ids, max_length=512, temperature=0.7, top_p=0.9)
  response = tokenizer.decode(output[0], skip_special_tokens=True)
  return response


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()

  try: 
    # Formatting as user input
    formatted_prompt = f"[INST] {prompt} [/INST]"
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
      output = model.generate(input_ids, max_length=512, temperature=0.7, top_p=0.9)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response
  
  except: 
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def ChatGPT_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  try: 
    # Formatting as user input
    formatted_prompt = f"[INST] {prompt} [/INST]"
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
      output = model.generate(input_ids, max_length=512, temperature=0.7, top_p=0.9)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response
  
  except: 
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"


def GPT4_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      # print ("---ashdfaf")
      # print (curr_gpt_response)
      # print ("000asdfhia")
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose: 
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()
  try: 
    # response = openai.Completion.create(
    #             model=gpt_parameter["engine"],
    #             prompt=prompt,
    #             temperature=gpt_parameter["temperature"],
    #             max_tokens=gpt_parameter["max_tokens"],
    #             top_p=gpt_parameter["top_p"],
    #             frequency_penalty=gpt_parameter["frequency_penalty"],
    #             presence_penalty=gpt_parameter["presence_penalty"],
    #             stream=gpt_parameter["stream"],
    #             stop=gpt_parameter["stop"],)
    
    # Llama version
    formatted_prompt = f"[INST] {prompt} [/INST]"
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=gpt_parameter["max_tokens"],  # Equivalent to OpenAI's max_tokens
            temperature=gpt_parameter["temperature"],
            top_p=gpt_parameter["top_p"],
            do_sample=True,  # Ensures non-deterministic output (like OpenAI)
        )

    # Decode response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response
  
  except: 
    print ("TOKEN LIMIT EXCEEDED")
    return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt): 
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose: 
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response


def get_embedding(text):
  # Set the padding token to be the eos_token if not already set
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  
  # Tokenize the input text and get input IDs
  inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

  # Create an attention mask
  attention_mask = inputs['attention_mask'] if 'attention_mask' in inputs else None
      
  # Get the model's output (hidden states)
  with torch.no_grad():
    outputs = model(**inputs, attention_mask=attention_mask, output_hidden_states=True)

  # Extract the hidden states from the outputs
  hidden_states = outputs.hidden_states
  
  # Use the last hidden state for embeddings
  # We take the mean of the last hidden state across tokens to get a sentence-level embedding
  embeddings = hidden_states[-1].mean(dim=1).squeeze().numpy()
  
  return embeddings


if __name__ == '__main__':
  # gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50, 
  #                  "temperature": 0, "top_p": 1, "stream": False,
  #                  "frequency_penalty": 0, "presence_penalty": 0, 
  #                  "stop": ['"']}
  gpt_parameter = {
        "temperature": 0.7,
        "max_tokens": 256,
        "top_p": 0.9,
        "frequency_penalty": 0.0,  # No direct equivalent in HF models
        "presence_penalty": 0.0,  # No direct equivalent in HF models
        "stream": False,  # Streaming is handled differently in HF
        "stop": None  # Stop tokens can be manually set if needed
    }
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)




















