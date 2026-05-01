# How to run the Project
Follow the steps below to set up and run the Histogram-Based Fake Image Detection System on your local machine.

1. Clone the Repository <br/>
git clone https://github.com/ayushmallah/Histogram-Based-Fake-Image-Detection-and-Performance-Analysis.git

2. Install Dependencies <br/>
pip install -r requirements.txt
<br/> &emsp; &emsp; &emsp; &emsp; or
<br/>pip3 install -r requirements.txt

3. Project Structure
Ensure your Project folder looks like this:<br/>
│── main.py<br/>
│── model.pkl<br/>
│── requirements.txt<br/>
├── dataset/<br/>
&emsp;&emsp;&emsp;&emsp;├── fake/images<br/>
&emsp;&emsp;&emsp;&emsp;└── real/images<br/>

4. Update Dataset Path
Open gui.py and make sure dataset paths are correct:<br/>
fake_path = "dataset/fake"<br/>
real_path = "dataset/real"<br/>

5. Run the train_model.py first it will train on the images you provide.<br/>
python train_model.py
<br/>&emsp;&emsp;&emsp;&emsp;or
<br/>python3 train_model.py

6. Run the main file:
<br/> python3 gui.py

7. How to Use
- Click Upload
-  Click Detect
- View:
   * Prediction (Real/Fake)
   * Confidence Score
   * Image Analysis (mean, noise, edges, etc.)
- Click Download Report ---> Save Pdf Report

8. Output Files
* Generated Images --> generated/ folder
* PDF Report --> User-selected location
* Metrics(optional) --> metrics.json

Requirements
* Python 3.8+
* Windows / macOS / Linux
* Minimum 4GB RAM recommended
* Dataset --> https://www.kaggle.com/datasets/shivamardeshna/real-and-fake-images-dataset-for-image-forensics
* Model file --> https://drive.google.com/file/d/12sWwT-hGFVtv0Fe8dRSeKK1eS2jfEFam/view?usp=drive_link
* Small dataset file --> https://drive.google.com/drive/folders/1r3Dquzy1OxFCUAeMIdbheuPs6WhZEcqj?usp=drive_link
