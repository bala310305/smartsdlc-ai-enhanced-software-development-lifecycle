# ✅ SmartSDLC – Requirement Extraction using AI (NLP)

!pip install transformers --quiet

from transformers import pipeline

# Sample client statement (like in the planning phase)
requirements_text = """
We are building an e-commerce website. The users should be able to register, login, browse products,
add items to cart, and make payments. Admins must be able to manage product listings and track orders.
"""

# Define possible software features
features = [
    "User Registration",
    "User Login",
    "Product Browsing",
    "Add to Cart",
    "Online Payments",
    "Admin Product Management",
    "Order Tracking",
    "User Reviews"
]

# Load zero-shot classification model
classifier = pipeline("zero-shot-classification")

# Predict relevant features
result = classifier(requirements_text, features)

print("✅ Extracted Requirements using AI:\n")
for label, score in zip(result["labels"], result["scores"]):
    if score > 0.3:
        print(f"- {label} (confidence: {score:.2f})")
