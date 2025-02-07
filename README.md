# **👕 Fashion-MNIST Image Classifier **  

🚀 A deep learning-powered **image classification platform** built with **Flask, TensorFlow, and Docker** to classify clothing items from the **Fashion-MNIST dataset**. This project allows users to upload images and get real-time predictions via a web interface.  

---

## **📌 Features**  
✅ **Deep Learning Model** trained on the Fashion-MNIST dataset  
✅ **Flask API** for handling image uploads and classification  
✅ **User-Friendly Web UI** with **Tailwind CSS**  
✅ **Containerized with Docker** for easy deployment  
✅ **Preprocessing Pipeline** for image normalization  
✅ **Supports PNG, JPG, and JPEG image formats**  

---

## **📂 Project Structure**  

```
📦 fashion-mnist-classifier
│-- 📂 models/                 # Trained CNN model and preprocessed dataset
│-- 📂 uploads/                # Stores uploaded images (created dynamically)
│-- 📂 templates/              # HTML templates for the frontend
│-- 📜 flask_api.py            # Flask API for image classification
│-- 📜 model.py                # CNN Model training and clustering
│-- 📜 requirements.txt        # Required dependencies
│-- 📜 Dockerfile              # Docker configuration
│-- 📜 README.md               # Project documentation (you're reading it 😂😂!)
```

---

## **🛠 Setup & Installation**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/Alchemy4587/Artificial-Intelligence-Exam-for-Group-11
```

### **2️⃣ Install Dependencies**  
Ensure you have Python **3.10+** installed. Then, run:  
```bash
pip install -r requirements.txt
```
### **3️⃣ Steps in running the project**  
```
1. Place the datasets in the dataset folder (link to dataset https://www.kaggle.com/datasets/zalando-research/fashionmnist)
  2. Run dataLoading.py
  2. Run model.py 
  3. Run flask_api.py
```

### **3️⃣ Train the Model (Because I did not push the trained model)**  
If you want to retrain the model, execute:  
```bash
python model.py
```
The trained model will be saved in the `models/` directory.

---

## **🚀 Running the Flask App**  

### **Run Locally**  
```bash
python flask_api.py
```
- Open **http://127.0.0.1:5000/** in your browser.  
- Upload an image to classify it.  

### **Run with Docker**  
#### **1️⃣ Build the Docker Image**  
```bash
docker build -t fashion-mnist-app .
```
#### **2️⃣ Run the Container**  
```bash
docker run -p 5000:80 fashion-mnist-app
```
- The app will be available at **http://localhost:5000/**.  

---

## **🖼️ Usage Guide**  
1️⃣ **Upload an image** (PNG, JPG, or JPEG).  
2️⃣ Click **"Classify Image"**.  
3️⃣ View the **predicted category** and **confidence score**.  

Example Output:  
```json
{
  "category": "Sneaker",
  "confidence": 95.2
}
```

---

## **💡 Model Performance**  
📊 The trained **CNN model** achieved:  
- **Training Accuracy:** `89.39%`  
- **Validation Accuracy:** `90.44%`  
- **Final Test Accuracy:** `90.13%`  

✅ The model is highly accurate in classifying Fashion-MNIST images.  

---

## **📌 Technologies Used**  
- **🖥️ Backend:** Flask, TensorFlow, NumPy, Pillow  
- **🎨 Frontend:** HTML, Tailwind CSS, JavaScript  
- **🐳 Deployment:** Docker  
- **📊 Model Training:** TensorFlow/Keras, ImageDataGenerator  


## **🤝 Contributing**  
🔧 Want to improve this project? Feel free to fork and submit a PR!  

👋 **Happy Coding 😁😁!** 🚀
