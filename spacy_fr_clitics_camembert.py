#pip inslall spacy
#python -m spacy download fr_dep_news_trf

#1-spacy
import spacy
spacy.require_gpu()
nlp = spacy.load("fr_dep_news_trf")#French transformer pipeline (camembert-base): le meilleur pour tager les parties du discours!
import fr_dep_news_trf
nlp = fr_dep_news_trf.load()

#2-Données : textes contenant potentionnellement des clitiques objets
with open('spacy.txt') as f:
    contents = f.read()
doc = nlp(contents)

#3-Fonction pouir isoler les phrases contenant des clitiques objets
def clitique(doc):
    clitiques = []
    for sentence in doc.sents:
        for token in sentence:
            if (
                token.text not in ["Que", "que", "Qu'", "qu'", "ce", "autre"]
                and token.pos_ == "PRON"
                and token.dep_ in ["obj", "iobj", "expl:comp"]
            ):
                clitiques.append(sentence.text)
    return clitiques

sentences = clitique(doc)

#4-CamemBERT-based model 
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
#Modèle de prévision basé sur CamemBERT : juegements-bis.pt
model = CamembertForSequenceClassification.from_pretrained("jugements-bis.pt", return_dict=False, num_labels = 3)
tokenizer = CamembertTokenizer.from_pretrained('camembert-base', do_lower_case=True)

#5-Fonctions de prédiction
def preprocess(phrases, labels=None):
    encoded_batch = tokenizer.batch_encode_plus(phrases,
                                                truncation=True,
                                                pad_to_max_length=True,
                                                return_attention_mask=True,
                                                return_tensors = 'pt')
    if labels:
        labels = torch.tensor(labels)
        return encoded_batch['input_ids'], encoded_batch['attention_mask'], labels
    return encoded_batch['input_ids'], encoded_batch['attention_mask']
 
def predict(phrases, model=model):
    with torch.no_grad():
        model.eval()
        input_ids, attention_mask = preprocess(phrases)
        retour = model(input_ids, attention_mask=attention_mask)
        return torch.argmax(retour[0], dim=1)

#6-Une boucle pour fournir des prédictions : 0 (erreur), 1 (correct), 2 (répétition)
for exemple in sentences:
    result = predict([exemple])[0]
    if result == 0:
        status = "Phrase agrammaticale!"
    elif result == 1:
        status = "Phrase grammaticale!"
    else:
        status = "Répétition!"
    print(exemple)
    print(" ==>", status)
    print()