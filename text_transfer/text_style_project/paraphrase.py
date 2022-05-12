#!/usr/bin/env python3
import os
import sys
import argparse
import re
import torch

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GPT2TokenizerFast


def paraphrase(text, model, tokenizer):
    if text[-5:] != '<bos>':
        text = text + '<bos>'
    input = tokenizer.encode(text,return_tensors='pt').to(device=model.device)
    ## Beam search
    output = model.generate(input, max_new_tokens=input.shape[-1]*5,
                        num_beams=3, 
                        no_repeat_ngram_size=2,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        early_stopping=True)
    # output = model.generate(input, max_new_tokens=input.shape[-1]*2,
    #                 pad_token_id=tokenizer.eos_token_id,
    #                 eos_token_id=tokenizer.eos_token_id,
    #                 bos_token_id=tokenizer.bos_token_id,
    #                 do_sample=True,  
    #                 top_p=0.6, 
    #                 top_k=0,
    #                 early_stopping=True)
    output = tokenizer.decode(output[0].tolist(), skip_special_tokens=False)
    if output[-5:] == '<eos>':
        output = output[:-5]
    result = re.search(r"\<bos\>(.*)$", output)
    if result == None:
        print('Unable to extract sentence: ')
        print(output)
        return ''
    else:
        return result.group(1)

def batch_paraphrase(sentences, output_path,
                    model, tokenizer,
                    start=0, end=None, 
                    saving_interval = 10): 
    if end == None:
        end = len(sentences)

    if torch.cuda.is_available():
        model.to(torch.device('cuda'))
    else:
        print('WARNING: Not using GPU')
    
    result = []
    count = 0
    n_elapse = 0
    for sen in sentences[start: end]:
        result.append(paraphrase(sen, model, tokenizer))
        count += 1
        if count == saving_interval:
            with open(output_path, 'a') as f:
                content = "\n".join(result) + '\n'
                f.write(content)
            result = []
            count = 0
            n_elapse += 1
            print(f'saved {n_elapse*saving_interval} entries')
    if len(result) > 0:
        with open(output_path, 'a') as f:
            content = "\n".join(result) + '\n'
            f.write(content)

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model_checkpoint', help="checkpoint of model and tokenizer", type=dir_path,
                        default='')
    parser.add_argument('infile', help="Input text file", type=argparse.FileType('r'))
    parser.add_argument('-o', '--outfile', help="Output file",
                        default='./output.txt', type=str)
    parser.add_argument('-i', '--interval', help="Interval of saving", type=int,
                        default=100)
    parser.add_argument('-s', '--start', help="start of line", type=int,
                        default=0)
    parser.add_argument('-e', '--end', help="end of line", type=int,
                        default=0)

    args = parser.parse_args(arguments)
    
    # data = pd.read_csv(args.infile)['text'].to_list()
    # with open(args.infile) as f:
    #     data = f.read().split("\n")

    data = args.infile.read().split("\n")
    assert args.model_checkpoint != ''
    # model_checkpoint = "filco306/gpt2-base-style-paraphraser"
    # tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)
    tokenizer = GPT2TokenizerFast(tokenizer_file=os.path.join(args.model_checkpoint,"tokenizer.json"),
                                    vocab_file=os.path.join(args.model_checkpoint,"vocab.json"),
                                    merges_file=os.path.join(args.model_checkpoint,"merges.txt"),
                                    bos_token='<bos>',
                                    eos_token='<eos>')
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint)
    end_row = len(data)
    if args.end != 0:
        end_row = args.end
    batch_paraphrase(data, args.outfile, 
                model, tokenizer, start=args.start, end=end_row, saving_interval=args.interval)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
