import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Define core symptoms for each disease
core_symptoms = {
    'Avian Influenza (Bird Flu)': ['Respiratory distress', 'Swollen head', 'Decreased egg production', 'Sudden death'],
    'Newcastle Disease': ['Coughing', 'Twisting neck', 'Paralysis', 'Greenish diarrhea', 'Lethargy'],
    'Infectious Bronchitis': ['Nasal discharge', 'Sneezing', 'Wrinkled eggs', 'Watery eyes', 'Reduced feed intake'],
    'Marek’s Disease': ['Paralysis of legs or wings', 'Weight loss', 'Tumors in organs', 'Grey iris'],
    'Fowl Pox': ['Scabby lesions on comb', 'Scabby lesions on wattles', 'Scabby lesions on legs', 'Reduced appetite', 'Lethargy'],
    'Salmonellosis': ['Diarrhea', 'Lethargy', 'Decreased egg production', 'Weight loss'],
    'Coccidiosis': ['Bloody diarrhea', 'Weakness', 'Ruffled feathers', 'Weight loss'],
    'Mycoplasma Gallisepticum': ['Coughing', 'Swollen sinuses', 'Nasal discharge', 'Conjunctivitis'],
    'Infectious Coryza': ['Facial swelling', 'Foul-smelling nasal discharge', 'Sneezing', 'Respiratory distress'],
    'Aspergillosis': ['Gasping', 'Weight loss', 'Respiratory distress', 'Yellow nodules in lungs'],
    'Gumboro Disease (IBD)': ['Swollen cloaca', 'Diarrhea', 'Ruffled feathers', 'Immunosuppression'],
    'Avian Encephalomyelitis': ['Tremors', 'Leg weakness', 'Paralysis', 'Head shaking'],
    'Fowl Cholera': ['Swollen wattles', 'Diarrhea', 'Sudden death', 'Joint swelling'],
    'Egg Drop Syndrome (EDS)': ['Soft-shelled eggs', 'Reduced egg production', 'Pale eggs'],
    'Botulism': ['Paralysis', 'Limber neck', 'Difficulty swallowing', 'Sudden death'],
    'Bumblefoot': ['Swollen foot pad', 'Limping', 'Abscess formation'],
    'Vitamin A Deficiency': ['Swollen eyes', 'Pale comb', 'Nasal discharge', 'Poor feather quality'],
    'Vitamin D Deficiency (Rickets)': ['Soft bones', 'Lameness', 'Poor egg shell quality', 'Bowed legs'],
    'Fatty Liver Hemorrhagic Syndrome': ['Obesity', 'Sudden death', 'Enlarged liver', 'Pale comb'],
    'Worm Infestation (Roundworms)': ['Weight loss', 'Diarrhea', 'Poor feather condition', 'Worms in droppings'],
    'Blackhead Disease (Histomoniasis)': ['Yellow droppings', 'Lethargy', 'Liver damage'],
    'E. coli Infection': ['Diarrhea', 'Swollen abdomen', 'Reduced egg production'],
    'Gizzard Worms': ['Poor growth', 'Lethargy', 'Digestive issues'],
    'Crop Impaction': ['Swollen crop', 'Difficulty swallowing', 'Weight loss'],
    'Scaly Leg Mites': ['Thickened, crusty leg scales', 'Discomfort'],
    'Heat Stress': ['Panting', 'Spread wings', 'Lethargy'],
    'Hypothermia': ['Fluffed-up feathers', 'Reduced activity', 'Shivering'],
    'Egg Binding': ['Straining', 'Swollen abdomen', 'Lethargy'],
    'Avian Tuberculosis': ['Weight loss', 'Lethargy', 'Diarrhea'],
    'Lead Poisoning': ['Weakness', 'Drooping wings', 'Seizures'],
    'Frostbite': ['Blackened combs or wattles', 'Pain'],
    'Feather Picking': ['Bald spots', 'Bleeding', 'Stress'],
    'Molting Issues': ['Excessive feather loss', 'Stress'],
    'Respiratory Mycoplasmosis': ['Sneezing', 'Nasal discharge', 'Coughing'],
    'Sour Crop': ['Swollen, squishy crop', 'Bad odor'],
    'Bacterial Enteritis': ['Diarrhea', 'Dehydration', 'Weight loss'],
    'Ascites (Water Belly)': ['Swollen abdomen', 'Difficulty breathing'],
    'Fowl Typhoid': ['Lethargy', 'Green diarrhea', 'Pale comb'],
    'Capillariasis (Threadworm Infestation)': ['Weight loss', 'Anemia', 'Diarrhea'],
}

# Step 2: Create a function to generate augmented samples
def augment_samples(disease, core_symptoms, num_samples=15):
    """
    Generate augmented samples for a given disease.
    """
    augmented_samples = []
    all_symptoms = list(core_symptoms.values())  # Flatten the list of all symptoms
    all_symptoms = list(set([symptom for sublist in all_symptoms for symptom in sublist]))  # Unique symptoms
    
    for _ in range(num_samples):
        # Set core symptoms to 1
        sample = {symptom: 1 for symptom in core_symptoms[disease]}
        
        # Randomly set non-core symptoms to 1 or 0
        for symptom in all_symptoms:
            if symptom not in core_symptoms[disease]:
                sample[symptom] = np.random.choice([0, 1], p=[0.8, 0.2])  # 20% chance of being 1
        
        # Add the disease name
        sample['Disease Name'] = disease
        augmented_samples.append(sample)
    
    return pd.DataFrame(augmented_samples)

# Step 3: Generate augmented samples for each disease (only if the CSV file doesn't exist)
import os
if not os.path.exists('augmented_poultry_disease_dataset_fixed.csv'):
    augmented_data = []
    for disease in core_symptoms:
        augmented_data.append(augment_samples(disease, core_symptoms, num_samples=15))

    # Combine all augmented samples into a single DataFrame
    df_augmented = pd.concat(augmented_data, ignore_index=True)

    # Save the augmented dataset to a new CSV file
    df_augmented.to_csv('augmented_poultry_disease_dataset_fixed.csv', index=False)
    print("Augmented dataset saved to 'augmented_poultry_disease_dataset_fixed.csv'")
else:
    print("Augmented dataset already exists. Skipping generation.")

# Load the augmented dataset
df_augmented = pd.read_csv('augmented_poultry_disease_dataset_fixed.csv')

# Prepare the data
X = df_augmented.drop(columns=['Disease Name'])  # Features (all symptoms)
y = df_augmented['Disease Name']  # Labels (disease names)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=200, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=model.classes_))

# Step 5: Make predictions on a sample input
sample_symptoms = {
   "Gasping":1,
   "Weight loss": 1,
   "Yellow nodules in lungs": 1,
   "Respiratory distress":1



}


# Convert the sample input into a DataFrame
sample_df = pd.DataFrame([sample_symptoms])

# Ensure the columns match the training data
for column in X.columns:
    if column not in sample_df.columns:
        sample_df[column] = 0

# Reorder columns to match the training data
sample_df = sample_df[X.columns]

# Make a prediction
predicted_disease = model.predict(sample_df)
predicted_probabilities = model.predict_proba(sample_df)

# Display the predicted disease and probabilities
print(f"\nPredicted Disease: {predicted_disease[0]}")
print("\nPredicted Probabilities:")
for disease, prob in zip(model.classes_, predicted_probabilities[0]):
    print(f"{disease}: {prob * 100:.2f}%")

# Step 6: Handle overlapping symptoms
# Check if the top predictions have similar probabilities
top_n = 3  # Number of top predictions to consider
top_indices = np.argsort(predicted_probabilities[0])[-top_n:][::-1]  # Indices of top predictions
top_diseases = model.classes_[top_indices]
top_probs = predicted_probabilities[0][top_indices]

# If the top predictions have similar probabilities, ask for more information
if top_probs[0] - top_probs[1] < 0.1:  # Threshold for similarity
    print("\nThe model is unsure between the following diseases:")
    for disease, prob in zip(top_diseases, top_probs):
        print(f"{disease}: {prob * 100:.2f}%")
    print("\nPlease provide more symptoms or context to help differentiate.")
else:
    print("\nThe model is confident in its prediction.")
