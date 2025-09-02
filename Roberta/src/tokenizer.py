from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import dill
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import RobertaTokenizerFast, RobertaModel
from transformers import Trainer, TrainingArguments
from transformers import pipeline
import torch
import os
import shutil
print(torch.cuda.is_available())
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit

def train_tokenizer():

    # paths = [str(x) for x in Path("./data/text/").glob("*.txt")]
    paths = ["./data/allstrandinst.txt"]

    # # Initialize a tokenizer
    # tokenizer = ByteLevelBPETokenizer()
    # # Customize training
    # tokenizer.train(files=paths, vocab_size=50000, min_frequency=2, special_tokens=[
    #     "<s>",
    #     "<pad>",
    #     "</s>",
    #     "<unk>",
    #     "<mask>",
    # ])
    # tokenizer.save_model("BERT")
    
     
    model_dir='/home2/kyhe/workspace/sem2vec-BERT/BERT2'
    # if(os.path.exists(model_dir)):
    #     shutil.rmtree(model_dir)
    # os.mkdir(model_dir)
    tokenizer = Tokenizer(WordLevel())
    trainer = WordLevelTrainer(vocab_size=50000, min_frequency=10, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    tokenizer.pre_tokenizer = WhitespaceSplit()

    tokenizer.train(files=paths,trainer=trainer)

    tokenizer.model.save(model_dir)
    merges=open(model_dir+'/merges.txt','w')
    tokenizer.save(model_dir+'/vocab.json')
    

def load_tokenizer(debug=False):
    tokenizer = RobertaTokenizerFast(
        vocab_file="BERT2/vocab.json", merges_file="BERT2/merges.txt"
    )
    
    
    # tokenizer = RobertaTokenizerFast.from_pretrained("./BERT", max_len=512)
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)
    return tokenizer

if __name__ == "__main__":
    train_tokenizer()
    
    # tokenizer = load_tokenizer(True)
    # s = 'ldx ss (eax+constant) ebx3 j (eax3a+ebx3) #2 @'
    # # result = tokenizer.encode(s)
    # print(s)
    # batch_encoding=tokenizer(s)
    # print(batch_encoding.tokens())
    # print(tokenizer.encode(s))
    
    model_dir = '/home2/kyhe/workspace/sem2vec-BERT/BERT2'
    # tokenizer1 = RobertaTokenizerFast.from_pretrained('/home2/kyhe/workspace/sem2vec-BERT/BERT')
    tokenizer2 = RobertaTokenizerFast(tokenizer_file=model_dir+'/vocab.json')

    corpora_line="ldx ds (var+(constant*eax)) si ttt"

    batch_encoding=tokenizer2(corpora_line)
    print(batch_encoding.tokens())
    print(tokenizer2.encode(corpora_line))
