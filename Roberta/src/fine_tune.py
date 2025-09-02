from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import pickle
import math
import torch
import os


os.makedirs("./BERT-ft", exist_ok=True)

batch_size = 16  
num_epoch = 32

def load_data():

    with open("./data/pair_update/test.pkl", "rb") as f:
        test = pickle.load(f)
    with open("./data/pair_update/train.pkl", "rb") as f:
        train = pickle.load(f)
    

    test_samples = []
    for pair in test:
        label = float(pair[2]) 
        inp_exp = InputExample(texts=[pair[0][:510], pair[1][:510]], label=label)
        test_samples.append(inp_exp)
    
    train_samples = []
    for pair in train:
        label = float(pair[2])
        inp_exp = InputExample(texts=[pair[0][:510], pair[1][:510]], label=label)
        train_samples.append(inp_exp)
    
    print(f"Number of training: {len(train_samples)}, Number of testing: {len(test_samples)}")
    return train_samples, test_samples

def finetune():

    print("Load Model...")
    model = SentenceTransformer("./BERT") 
    
    print("Load Data...")
    train_samples, test_samples = load_data()
    
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
    
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        test_samples, 
        name='sts-dev',
        show_progress_bar=True
    )
    
    train_loss = losses.CosineSimilarityLoss(model=model)
    
    warmup_steps = math.ceil(len(train_dataloader) * num_epoch * 0.1)

    
    print("Finetune...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epoch,
        evaluation_steps=1000,  
        warmup_steps=warmup_steps,
        output_path="./BERT-ft",
        save_best_model=True,  
        show_progress_bar=True
    )
    
    print("Saved to ./BERT-ft")



if __name__ == "__main__":
    finetune()