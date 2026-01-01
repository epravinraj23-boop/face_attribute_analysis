import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("list_attr_celeba.csv")

print("Dataset Loaded Successfully")
print("Total Images:", df.shape[0])
print("Total Attributes:", df.shape[1] - 1)

# Convert -1 to 0
df.iloc[:, 1:] = df.iloc[:, 1:].replace(-1, 0)

# -----------------------------
# Gender Distribution
# -----------------------------
gender_counts = df['Male'].value_counts()

plt.figure()
gender_counts.plot(kind='bar')
plt.title("Gender Distribution")
plt.xlabel("Gender (0 = Female, 1 = Male)")
plt.ylabel("Count")
plt.show()

# -----------------------------
# Smiling Analysis by Gender
# -----------------------------
smile_gender = df.groupby('Male')['Smiling'].mean()

plt.figure()
smile_gender.plot(kind='bar')
plt.title("Smiling Probability by Gender")
plt.xlabel("Gender (0 = Female, 1 = Male)")
plt.ylabel("Probability")
plt.show()

# -----------------------------
# Eyeglasses Analysis
# -----------------------------
glasses_counts = df['Eyeglasses'].value_counts()

plt.figure()
glasses_counts.plot(kind='bar')
plt.title("Eyeglasses Distribution")
plt.xlabel("0 = No Glasses, 1 = Glasses")
plt.ylabel("Count")
plt.show()

# -----------------------------
# Beard Features (Males Only)
# -----------------------------
beard_features = ['Beard', 'Goatee', 'Mustache']
male_beard = df[df['Male'] == 1][beard_features].mean()

plt.figure()
male_beard.plot(kind='bar')
plt.title("Beard Feature Probability (Males)")
plt.ylabel("Probability")
plt.show()

# -----------------------------
# Top 10 Most Common Attributes
# -----------------------------
top_attributes = df.iloc[:, 1:].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 4))
top_attributes.plot(kind='bar')
plt.title("Top 10 Facial Attributes")
plt.ylabel("Probability")
plt.show()

# -----------------------------
# Attribute Correlation Heatmap
# -----------------------------
plt.figure(figsize=(10, 8))
corr = df.iloc[:, 1:21].corr()  # first 20 attributes
sns.heatmap(corr, cmap='coolwarm')
plt.title("Attribute Correlation Heatmap")
plt.show()

# -----------------------------
# DISPLAY SAMPLE FACE IMAGES
# -----------------------------
image_folder = "images"

if os.path.exists(image_folder):
    image_files = os.listdir(image_folder)[:5]

    plt.figure(figsize=(10, 4))

    for i, img_name in enumerate(image_files):
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(1, 5, i + 1)
            plt.imshow(img)
            plt.axis("off")

    plt.suptitle("Sample Face Images from Dataset")
    plt.show()
else:
    print("Images folder not found!")

# -----------------------------
# SUMMARY INSIGHTS
# -----------------------------
print("\n----- DATA INSIGHTS -----")
print("Male Percentage:", gender_counts.get(1, 0) / len(df) * 100)
print("Female Percentage:", gender_counts.get(0, 0) / len(df) * 100)
print("People Smiling (%):", df['Smiling'].mean() * 100)
print("People Wearing Glasses (%):", df['Eyeglasses'].mean() * 100)
