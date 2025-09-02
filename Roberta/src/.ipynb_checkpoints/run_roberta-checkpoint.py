from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import RobertaTokenizerFast, RobertaModel
from transformers import Trainer, TrainingArguments
from transformers import pipeline
# import tensorflow as tf
import numpy as np

def train():
    config = RobertaConfig(
        vocab_size=50000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    # tokenizer = RobertaTokenizerFast(vocab_file="BERT2/vocab.json", merges_file="BERT2/merges.txt")
    tokenizer = RobertaTokenizerFast(tokenizer_file='BERT2/vocab.json')
    model = RobertaForMaskedLM(config=config)
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="./data/strandinst.txt",
        block_size=128,
    )

    data_collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer, mlm_probability=0.15
    )
    training_args = TrainingArguments(
        output_dir="./BERT2",
        overwrite_output_dir=True,
        num_train_epochs=512,
        per_device_train_batch_size=64,
        save_steps=10000,
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

    trainer.save_model("./BERT2")


def eval():

    # embed before finetuning
    
#     tokenizer = RobertaTokenizerFast.from_pretrained("./strandBERT")
#     model = RobertaModel.from_pretrained("./strandBERT")
    

#     inputs = tokenizer(["setz icall cs,[cs:([cs:var10]+#8)]<fast:_DWORD var10>, #0, dc6", 'setz icall cs,[cs:([cs:var10]+#8)]<fast:_DWORD var10>, #0, dc6'], return_tensors="pt") # padding=True, truncation=True, 
#     outputs = model(**inputs)

#     last_hidden_states = outputs.last_hidden_state.detach().numpy()
#     print(last_hidden_states)
    
    
    
#     inputs = tokenizer(["setz icall cs,[cs:([cs:var10]+#8)]<fast:_DWORD var10>, #0, dc6setz icall cs,[cs:([cs:var10]+#8)]<fast:_DWORD var10>, #0, dc6"], return_tensors="pt") # padding=True, truncation=True, 
#     outputs = model(**inputs)

#     last_hidden_states = outputs.last_hidden_state.detach().numpy()
#     print(last_hidden_states)
   
    # print(tokenizer.tokenize("setz icall cs,[cs:([cs:var10]+#8)]<fast:_DWORD var10>, #0<mask>, dc6"))
    

    
    fill_mask = pipeline(
        "fill-mask",
        model="./BERT/checkpoint-1600000",
        # tokenizer="./BERT/checkpoint-1600000"
        tokenizer=RobertaTokenizerFast(tokenizer_file='BERT/vocab.json')
    )
    
    formula = "ldx ds (<mask>ar+(constant*eax))"
    for f in fill_mask(formula):
        print(f)
        
    


if __name__ == "__main__":
    # train()
    # eval()
