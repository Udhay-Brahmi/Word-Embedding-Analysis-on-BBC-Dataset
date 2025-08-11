# 📊 Word Embeddings Evaluation Project

## 🎯 Project Overview

This project presents a comprehensive comparative analysis of different word embedding techniques trained on the BBC dataset, evaluated against GloVe embeddings using cosine similarity and Euclidean distance metrics.

### 👥 Team Members
- **23M2107** (2026, CSE Department)
- **23M0762** (2025, CSE Department)  
- **210100050** (2025, Mechanical Department)

---

## 🎲 Problem Statement

**Evaluation of different word embedding techniques (FastText, Skip-gram, CBOW, LSA) trained on BBC dataset against GloVe by cosine similarity and Euclidean distance.**

---

## 💡 Motivation

- Word embeddings play a **pivotal role** in the effective training and working of Large Language Models
- FastText is widely utilized but often requires substantial data volumes for optimal performance
- This project implements various word embedding techniques on a **smaller dataset** (BBC corpus) for comparative analysis against GloVe
- Addresses the challenge of training effective embeddings with limited data resources

---

## 📊 Dataset Information

### BBC Dataset Specifications
- **Total Records**: 2,225 sentences
- **Categories**: 5 main categories
  - Technology
  - Business  
  - Sports
  - Entertainment
  - Politics
- **Source**: Available on Hugging Face
- **Size**: Approximately 26,283 words (compact corpus for focused analysis)

---

## 🛠️ Implemented Techniques

### 1. 🚀 FastText
- **Approach 1**: Using built-in FastText functions
- **Approach 2**: Custom implementation without FastText libraries
- **Parameters**: 
  - Vector size: 100/300 dimensions
  - Window size: 5
  - Min count: 3
  - Workers: 4
  - Epochs: 10-50

### 2. 🎯 Skip-gram Model
- **Architecture**: Neural network with embedding layer
- **Training Data**: 108,016 word-context pairs
- **Vocabulary Size**: 5,368 unique words
- **Embedding Dimension**: 100
- **Training Epochs**: 15
- **Final Training Accuracy**: ~9.33%

### 3. 📍 CBOW (Continuous Bag of Words)
- **Architecture**: Context-to-word prediction
- **Training Data**: 29,158 context-target pairs
- **Window Size**: 4 (2 words on each side)
- **Embedding Dimension**: 100
- **Training Epochs**: 50
- **Final Training Accuracy**: ~87.07%

### 4. 📈 LSA (Latent Semantic Analysis)
- **Method**: SVD on PPMI (Positive Pointwise Mutual Information) matrices
- **Dimensionality Reduction**: Compact meaningful representations
- **Advantage**: Handles sparse data effectively

### 5. 🌍 GloVe (Baseline)
- **Purpose**: Reference model for comparison
- **Type**: Global vector representation
- **Training**: Pre-trained embeddings

---

## 🔬 Evaluation Methodology

### Core Function Implementation
```python
k_nearest_words(word_or_embedding, k)
```
Returns the k-nearest words for a given word or embedding using:

### Distance Metrics

#### 🎯 Cosine Similarity
- **Scale-invariant**: Unaffected by vector magnitude
- **Angle-based**: Measures direction similarity
- **High-dimensional friendly**: Effective in sparse spaces
- **Range**: [-1, 1] where 1 = identical direction

#### 📏 Euclidean Distance  
- **Magnitude consideration**: Accounts for both direction and magnitude
- **Intuitive interpretation**: Geometric distance in space
- **Widely applicable**: Works across various data types

---

## 📋 Implementation Pipeline

### Training Process
1. **Data Preprocessing**: Tokenization and vocabulary building
2. **Model Training**: Individual training for each embedding technique
3. **Vector Extraction**: Export trained embeddings
4. **Evaluation Setup**: k-nearest neighbor similarity calculations
5. **Comparative Analysis**: Cross-model performance comparison

### Key Implementation Steps
- Concatenate all models for easy iteration
- Extract embeddings from each trained model
- Create analogy test sets with word tuples
- Compute embeddings for analogy evaluation
- Apply cosine similarity for nearest neighbor search
- Compare predictions against ground truth

---

## 📊 Results & Performance

### Model Comparison Highlights

#### Skip-gram Performance
- **Strength**: Effective for rare words and contextual relationships
- **Training Accuracy**: 9.33% (15 epochs)
- **Best Analogy**: Limited success on small corpus
- **Memory**: High requirements for dense vectors

#### CBOW Performance  
- **Strength**: Faster training, stable with smaller datasets
- **Training Accuracy**: 87.07% (50 epochs) 
- **Efficiency**: Lower computational requirements
- **Trade-off**: Less effective for rare words

#### FastText Results
- **Analogy Success**: Predicted "smaller" for "small is to smaller"
- **King-Queen Relationship**: Successfully predicted "king" for gender analogy
- **OOV Handling**: Better performance with unseen words

#### LSA Findings
- **Analogy Performance**: Predicted "bigger" for "small is to smaller as large is to"
- **Semantic Capture**: Showed understanding of comparative relationships
- **Efficiency**: Good performance on reduced corpus size

### Similarity Scores
```python
# Example similarity measurements
cosine_similarity([skipgram['king']], [skipgram['queen']]) = 0.357
cosine_similarity([cbow['king']], [cbow['queen']]) = 0.320
```

---

## 🔍 Key Findings

### ✅ Successes
- **FastText** showed slight superiority in analogy tasks
- **CBOW** achieved highest training accuracy (87.07%)
- **LSA** demonstrated semantic understanding despite statistical approach
- All models successfully generated meaningful embeddings from limited data

### 🚨 Limitations
- **Corpus Size**: 26,283 words insufficient for robust analogy performance
- **Analogy Accuracy**: No model achieved perfect analogy predictions
- **Data Dependency**: Larger corpus needed for definitive model ranking
- **Training Complexity**: Balance between model sophistication and data requirements

### 📈 Comparative Insights
- **FastText**: Most promising for small corpus applications
- **CBOW**: Best training stability and efficiency
- **Skip-gram**: Superior contextual understanding potential
- **LSA**: Competitive performance with lower computational requirements

---

## 🔧 Technical Requirements

### Dependencies
```python
# Core libraries
pandas, numpy, nltk
gensim, sklearn
tensorflow/keras
matplotlib, seaborn

# Specific requirements
FastText, Word2Vec
TensorFlow 2.x
Scikit-learn
```
---

## 🚀 Usage Instructions

### 1. Data Preparation
```python
df = pd.read_csv('bbc-text.csv')
sentences = preprocess_text(df['text'])
```

### 2. Model Training
```python
# FastText
model = FastText(sentences, vector_size=100, window=5, min_count=3)

# Skip-gram/CBOW
model = Word2Vec(sentences, sg=1, vector_size=100, window=5)
```

### 3. Evaluation
```python
# Find similar words
similar_words = k_nearest_words('king', k=10)
```

---

## 📚 References

1. **Enriching Word Vectors with Subword Information** - FastText methodology
2. **Efficient Estimation of Word Representations in Vector Space** - Skip-gram & CBOW
3. **GloVe: Global Vectors for Word Representation** - GloVe implementation
4. **Latent Semantic Analysis Approach** - LSA for document summarization
5. **Evaluating Word Embedding Models** - Evaluation metrics and methodologies

---

*"Despite the challenges of working with a limited corpus, this project demonstrates the feasibility of training meaningful word embeddings on smaller datasets, with FastText showing particular promise for resource-constrained scenarios."*
