from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Sentences to compare
sentence1 = "What is Artificial Intelligence?"
sentence2 = "Explain the concept of machine learning."

# Get embeddings for the sentences
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)

# Calculate the cosine similarity between the two embeddings
similarity = cosine_similarity([embedding1], [embedding2])

# Print the similarity score
print("Cosine Similarity between the sentences:", similarity[0][0])
