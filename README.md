#  Contextual Fraud Detection with Generative AI

This project explores how Generative AI can improve credit card fraud detection by adding human-readable, neutral, and informative context to transactional datasets. Using a combination of **GPT-4o** and **T5 summarization**, we create augmented features that help logistic regression models perform better on fraud classification tasks.

---

##  Project Structure
```
contextual-fraud-detection-genai/
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_augmentation_textual_generation.ipynb
│   └── 03_model_training_evaluation.ipynb
│
├── data/
│   └── final_dataset_with_context.csv
│
├── README.md
```
---

##  Key Steps in Methodology

1. **Context Generation**: GPT-4o was used to generate transaction descriptions.
2. **Bias Neutralization**: Regex was applied to remove labels like “fraudulent” or “legit”.
3. **Summarization**: Texts were summarized using HuggingFace’s T5-small model.
4. **TF-IDF + One-Hot Encoding**: Combined textual and categorical features.
5. **Logistic Regression**: Trained with L1 (sparse) and L2 (ridge) regularization.
6. **Evaluation**: Used precision, recall, F1-score and cross-validation.

---

##  Results Summary

- **Best F1-Score**: 0.978 with **L1 Regularization**
- **Cross-Validation Mean Accuracy**: 0.9707
- **Balanced Performance** across both fraud and non-fraud classes

---

##  Tools Used

- Python (Pandas, Scikit-learn, Regex, SciPy, TF-IDF)
- OpenAI GPT-4o (for contextual generation)
- Hugging Face (T5-small summarizer)
- SMOTE (class balancing)
- Jupyter Notebooks

---


## Contact

For questions or collaboration, feel free to connect with [Pranav CR](https://github.com/PranavCR01).
