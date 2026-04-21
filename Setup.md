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
│
├── dataset/<br/>
│ &emsp;&emsp;&emsp;&emsp;├── fake/images<br/>
│ &emsp;&emsp;&emsp;&emsp;└── real/images<br/>

4. Update Dataset Path
Open gui.py and make sure dataset paths are correct:<br/>
fake_path = "dataset/fake"<br/>
real_path = "dataset/real"<br/>

5. Run the train_model.py first it will train on the images you provide.<br/>
python train_model.py
<br/>&emsp;&emsp;&emsp;&emsp;or
<br/>&emsp;&emsp;&emsp;&emsp;python3 train_model.py

6. 
