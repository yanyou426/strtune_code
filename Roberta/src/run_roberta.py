# -*- coding: utf-8 -*-
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import RobertaTokenizerFast, RobertaModel
from transformers import Trainer, TrainingArguments
# from transformers import pipeline
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import os
import numpy as np

def create_word_level_tokenizer(file_path, vocab_size=50000):
    
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    

    from tokenizers.trainers import WordLevelTrainer
    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )
    
    tokenizer.train(files=[file_path], trainer=trainer)
    

    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )
    
    return tokenizer

def train():

    os.makedirs("./BERT", exist_ok=True)
    

    tokenizer = create_word_level_tokenizer("./data/strandinst.txt")
    

    tokenizer.save("./BERT/tokenizer.json")
    

    config = RobertaConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    
    dataset = LineByLineTextDataset(
        tokenizer=RobertaTokenizerFast(tokenizer_object=tokenizer),
        file_path="./data/strandinst.txt",
        block_size=128,
    )


    data_collator = DataCollatorForWholeWordMask(
        tokenizer=RobertaTokenizerFast(tokenizer_object=tokenizer), 
        mlm_probability=0.15
    )
    

    model = RobertaForMaskedLM(config=config)
    

    training_args = TrainingArguments(
        output_dir="./BERT",
        overwrite_output_dir=True,
        num_train_epochs=10,  
        per_device_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

    trainer.save_model("./BERT")
    tokenizer.save("./BERT/tokenizer.json")



if __name__ == "__main__":
    train()
