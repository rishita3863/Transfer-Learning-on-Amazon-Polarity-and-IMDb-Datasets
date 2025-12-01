# Transfer-Learning-on-Amazon-Polarity-and-IMDb-Datasets
A model has been built based on Amazon Polarity dataset and the same model has been fine tuned and applied to IMDb dataset. Transfer learning concept has been applied.

````markdown
<div align="center">

# 🚀 Sentiment Analysis using RoBERTa  
### Amazon Reviews Polarity → IMDB Transfer Learning  

![GitHub Repo Size](https://img.shields.io/github/repo-size/yourusername/yourrepo)
![GitHub Stars](https://img.shields.io/github/stars/yourusername/yourrepo?style=social)
![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![HuggingFace](https://img.shields.io/badge/Transformers-🤗-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A complete NLP workflow:  
Train **RoBERTa** on Amazon Reviews → Test on IMDB → Fine-tune → Visualize metrics.

</div>

---

# 📌 **Overview**

This project demonstrates how a transformer model can learn sentiment from one domain (Amazon shopping reviews) and transfer to another domain (IMDB movie reviews).

### ✔ Features
- Fine-tune **RoBERTa-base** on Amazon Reviews Polarity  
- Evaluate on IMDB (cross-domain testing)  
- Fine-tune further on IMDB  
- Plot **training/validation** curves  
- Extract metrics from **HuggingFace Trainer logs**  
- GPU-optimized training (Google Colab)

---

# 🧠 **Model Architecture**

<div align="center">

### **RoBERTa-base (125M parameters)**

Tokenization → Embeddings → 12 Transformer Layers → Classification Head

<img src="https://huggingface.co/front/thumbnails/roberta-base.png" width="500"/>

</div>

---

# 📊 **Datasets**

## **1. Amazon Reviews Polarity**

* Binary sentiment (Positive/Negative)
* Large, clean, short reviews
* Perfect for initial fine-tuning

## **2. IMDB Movie Reviews**

* Binary sentiment
* Longer and more expressive
* Used for testing domain transfer
* Fine-tuned further if accuracy is low

---

# 🔧 **Setup**

### **Install Dependencies**

```bash
pip install transformers datasets accelerate torch matplotlib
```

### **Import Libraries**

```python
from datasets import load_dataset
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
```

---

# 🏋️‍♂️ **Training the Amazon Model**

Train RoBERTa-base for sentiment classification:

```python
amazon_dataset = load_dataset("amazon_polarity")
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
```

Training handled by HuggingFace Trainer:

```python
training_args = TrainingArguments(
    output_dir="amazon_model",
    num_train_epochs=15,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True
)

amazon_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=amazon_train,
    eval_dataset=amazon_test,
    tokenizer=tokenizer
)

amazon_trainer.train()
```

---

# 🧪 **Zero-Shot Evaluation on IMDB**

```python
imdb = load_dataset("imdb")
results = amazon_trainer.evaluate(imdb_test_tokenized)
print(results)
```

If accuracy is low → proceed to fine-tuning.

---

# 🔄 **Fine-Tuning on IMDB**

```python
training_args = TrainingArguments(
    output_dir="imdb_model",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    fp16=True
)

imdb_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=imdb_train,
    eval_dataset=imdb_test,
    tokenizer=tokenizer
)

imdb_trainer.train()
```

---

# 📈 **Training & Validation Curves**

You can recover metrics directly from Trainer logs:

```python
history = amazon_trainer.state.log_history
```

### **Example Plot Output**

(Replace with your actual PNG files)

<div align="center">

### Amazon Training Curves

<img src="plots/amazon_loss.png" width="400"/>
<img src="plots/amazon_accuracy.png" width="400"/>

### IMDB Fine-Tuning Curves

<img src="plots/imdb_loss.png" width="400"/>
<img src="plots/imdb_accuracy.png" width="400"/>

</div>

---

# 📌 **Results**

| Model                          | Dataset         | Accuracy                           |
| ------------------------------ | --------------- | ---------------------------------- |
| RoBERTa-base (fine-tuned)      | Amazon Polarity | **96%**                            |
| Zero-Shot                      | IMDB            | Lower due to domain shift          |
| RoBERTa-base (fine-tuned IMDB) | IMDB            | **High accuracy after adaptation** |

---

# 🧭 **Key Learnings**

* Transformer models generalize well but require domain adaptation
* Amazon → IMDB transfer shows real-world distribution shift
* Fine-tuning restores performance quickly
* Trainer logs are extremely useful for visualizing learning dynamics

---

# 🔮 **Future Enhancements**

* Upgrade to **RoBERTa-large** or **DeBERTa-v3**
* Add **Optuna hyperparameter search**
* Use **W&B or MLflow** for experiment tracking
* Add **early stopping** & **LR schedulers**

---


---

# 🤝 **Contributions**

Pull requests welcome!
Feel free to open issues or suggest new features.

---

# 🌟 If this repo helped you, consider giving it a star!



