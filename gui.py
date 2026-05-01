import json
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from reportlab.platypus import Image as RLImage
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

metrics_loaded_once = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_FILE = os.path.join(BASE_DIR, "metrics.json")

def get_dataset_info():
    fake_path = "fake images dataset path"
    real_path = "real images dataset path"

    valid_extensions = (".png", ".jpg", ".jpeg")

    fake_count = len([f for f in os.listdir(fake_path) if f.lower().endswith(valid_extensions)])
    real_count = len([f for f in os.listdir(real_path) if f.lower().endswith(valid_extensions)])

    return fake_count, real_count, fake_count + real_count


# cache function
def save_metrics(metrics):
    metrics["timestamp"] = time.ctime()
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f)

def load_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return None




def generate_pdf_report(
    image_path,
    result,
    confidence,
    mean,
    std,
    gaps,
    edge_density,
    noise_level,
    real_prob,
    fake_prob,
):

    global metrics_loaded_once

    if not metrics_loaded_once:
        metrics = load_metrics()
        
        if metrics is None:
            metrics = {
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "specificity": 0,
        "f1": 0,
        "auc": 0,
        "timestamp": "Not Available"
        }
        
        
        metrics_loaded_once = True
    else:
        metrics = load_metrics()

    # 🔹 Ask where to save PDF
    file_path = filedialog.asksaveasfilename(
        defaultextension=".pdf", filetypes=[("PDF File", "*.pdf")]
    )

    if not file_path:
        return

    # 🔹 Setup PDF
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # 🔹 Title
    elements.append(Paragraph("<b>Image Detection Report</b>", styles["Title"]))
    elements.append(Spacer(1, 20))

    # 🔹 Image
    elements.append(Paragraph("<b>Input Image:</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    elements.append(RLImage(image_path, width=300, height=200))
    elements.append(Spacer(1, 20))

    # 🔹 Result Section
    elements.append(Paragraph("<b>Prediction Result</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    result_data = [
        ["Result", result],
        ["Confidence", f"{confidence:.2f}%"]
    ]

    table = Table(result_data)
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ])
    )

    elements.append(table)
    elements.append(Spacer(1, 20))

    # 🔹 Analysis Section
    elements.append(Paragraph("<b>Image Analysis</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    analysis_data = [
        ["Mean", f"{mean:.4f}"],
        ["Std Deviation", f"{std:.4f}"],
        ["Histogram Gaps", str(gaps)],
        ["Edge Density", f"{edge_density:.4f}"],
        ["Noise Level", f"{noise_level:.4f}"],
    ]

    table2 = Table(analysis_data)
    table2.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))

    elements.append(table2)
    elements.append(Spacer(1, 20))

    # 🔹 Probabilities Section
    elements.append(Paragraph("<b>Model Probabilities</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    prob_data = [
        ["Real Probability", f"{real_prob:.2f}%"],
        ["Fake Probability", f"{fake_prob:.2f}%"],
    ]

    table3 = Table(prob_data)
    table3.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))

    elements.append(table3)
    elements.append(Spacer(1, 20))

    # 🔹 Interpretation
    elements.append(Paragraph("<b>Interpretation</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    interpretation_text = """
    High noise levels and irregular edge density may indicate possible image manipulation.
    Histogram gaps suggest unnatural pixel distribution often found in edited images.
    """

    elements.append(Paragraph(interpretation_text, styles["Normal"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("<b>Feature Engineering Explanation</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    feature_text = """
    This model analyzes multiple image characteristics:<br/><br/>

    • Histogram Distribution: Detects unnatural pixel intensity patterns.<br/>
    • Mean & Standard Deviation: Measures brightness and contrast variations.<br/>
    • Histogram Gaps: Identifies missing intensity values common in edited images.<br/>
    • Edge Density: Fake images often have irregular edge structures.<br/>
    • Noise Level: Manipulated images show abnormal noise patterns.
    """

    elements.append(Paragraph(feature_text, styles["Normal"]))
    elements.append(Spacer(1, 20))

    # 🔹 Model Performance Section
    elements.append(Paragraph("<b>Model Performance</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    try:
        perf_data = [
            ["Accuracy", f"{metrics.get('accuracy', 0):.4f}"],
            ["Precision", f"{metrics.get('precision', 0):.4f}"],
            ["Recall", f"{metrics.get('recall', 0):.4f}"],
            ["Specificity", f"{metrics.get('specificity', 0):.4f}"],
            ["F1 Score", f"{metrics.get('f1', 0):.4f}"],
            ["AUC", f"{metrics.get('auc', 0):.4f}"],
            ["Last Updated", metrics.get("timestamp", "N/A")],
        ]

        table4 = Table(perf_data)
        table4.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))

        elements.append(table4)
        elements.append(Spacer(1, 20))

        # 🔹 Add ROC Curve Image
        roc_path = os.path.join(os.path.dirname(__file__), "roc_curve.png")
        if os.path.exists(roc_path):
            elements.append(Paragraph("<b>ROC Curve</b>", styles["Heading2"]))
            elements.append(Spacer(1, 10))
            elements.append(RLImage(roc_path, width=300, height=200))

        cm_path = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")

        if os.path.exists(cm_path):
            elements.append(Paragraph("<b>Confusion Matrix</b>", styles["Heading2"]))
            elements.append(Spacer(1, 10))
            elements.append(RLImage(cm_path, width=300, height=200))
            elements.append(Spacer(1, 20))

            # 🔹 Explanation (VERY IMPORTANT FOR VIVA)
            cm_text = """
            TP (True Positive): Correctly predicted real images.<br/>
            TN (True Negative): Correctly predicted fake images.<br/>
            FP (False Positive): Fake image predicted as real.<br/>
            FN (False Negative): Real image predicted as fake.
            """

            elements.append(Paragraph(cm_text, styles["Normal"]))
            elements.append(Spacer(1, 20))

    except Exception as e:
        elements.append(Paragraph("Performance data unavailable", styles["Normal"]))

    # 🔹 Dataset Info
    elements.append(Paragraph("<b>Dataset Summary</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    fake_count, real_count, total = get_dataset_info()

    dataset_data = [
        ["Total Images", total],
        ["Fake Images", fake_count],
        ["Real Images", real_count],
    ]

    table_ds = Table(dataset_data)
    table_ds.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))

    elements.append(table_ds)
    elements.append(Spacer(1, 20))

    # 🔹 Build PDF
    doc.build(elements)


def download_report():

    generate_pdf_report(
        imagePath,
        last_result,
        last_confidence,
        last_mean,
        last_std,
        last_gaps,
        last_edge,
        last_noise,
        last_real_prob,
        last_fake_prob,
    )


def imageUpload():
    global imagePath

    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    imagePath = filedialog.askopenfilename(filetypes=fileTypes)
    actionButton.config(text="DETECT", command=imageDetect)
    if len(imagePath):
        img = Image.open(imagePath)
        img = img.resize((400, 400))
        pic = ImageTk.PhotoImage(img)
        upload.place(x=535, y=700)
        database_info.place(x=535, y=800)
        actionButton.place(x=835, y=700)
        generate_btn.place(x=835, y=800)
        uploadedImage.config(image=pic)
        uploadedImage.image = pic
        label.place_forget()


def extractImageData(image_path):
    import cv2
    import numpy as np

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error loading image")

    image = cv2.resize(image, (256, 256))

    # RGB Histograms
    hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])

    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean = np.mean(gray)
    std = np.std(gray)

    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    gaps = np.sum(hist_gray == 0) / 256

    # Edge
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (256 * 256)

    # Noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray - blur
    noise_level = np.std(noise)

    features = np.hstack([
        hist_r, hist_g, hist_b,
        mean, std, gaps,
        edge_density, noise_level
    ])

    # 🔥 RETURN ALL VALUES (IMPORTANT)
    return (
        np.array(features),
        hist_gray,
        mean,
        std,
        gaps,
        edge_density,
        noise_level
    )


def imageDetect():
    if not imagePath:
        return

    features, hist, mean, std, gaps, edge_density, noise_level = extractImageData(
        imagePath
    )

    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)
    prob = model.predict_proba(features)

    confidence = np.max(prob) * 100

    # Correct probability mapping
    fake_prob = prob[0][0] * 100
    real_prob = prob[0][1] * 100

    # Correct label mapping
    if prediction[0] == 1:
        result_label.config(text=f"Real Image ✅ ({confidence:.2f}%)", fg="green")
    else:
        result_label.config(text=f"Fake Image ❌ ({confidence:.2f}%)", fg="red")

    # Store values globally (or pass directly)
    global last_result, last_confidence, last_mean, last_std, last_gaps
    global last_edge, last_noise, last_real_prob, last_fake_prob

    last_result = "Real" if prediction[0] == 1 else "Fake"
    last_confidence = confidence
    last_mean = mean
    last_std = std
    last_gaps = gaps
    last_edge = edge_density
    last_noise = noise_level
    last_real_prob = real_prob
    last_fake_prob = fake_prob

    analysis_text = f"""
Mean: {mean:.4f}
Std Dev: {std:.4f}
Histogram Gaps: {gaps}

Edge Density: {edge_density:.4f}
Noise Level: {noise_level:.4f}

Real Probability: {real_prob:.2f}%
Fake Probability: {fake_prob:.2f}%
"""
    
    generate_variations(imagePath)

    result_label.place(x=630, y=150)
    analysis_label.config(text=analysis_text)
    analysis.place(x=700, y=50)
    actionButton.config(text="Download Report", command=download_report)

    show_histogram(imagePath)


def show_histogram(image_path):
    for widget in graph_frame.winfo_children():
        widget.destroy()

    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))

    fig, ax = plt.subplots(figsize=(5, 5))

    colors = ("BLUE", "GREEN", "RED")
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist = np.log1p(hist.flatten())
        ax.bar(range(256), hist, color=color, label=color.upper())

    ax.set_title("Log RGB Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Log Frequency")
    ax.legend()

    # Embed into Tkinter
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    plt.close(fig)


def database_information():

    fakeImagesPath = "fake images dataset path"
    realImagesPath = "real images dataset path"

    numberOfFakeFiles = len(
        [
            f
            for f in os.listdir(fakeImagesPath)
            if os.path.isfile(os.path.join(fakeImagesPath, f))
        ]
    )
    numberOfRealFiles = len(
        [
            f
            for f in os.listdir(realImagesPath)
            if os.path.isfile(os.path.join(realImagesPath, f))
        ]
    )

    dataSize = [numberOfFakeFiles, numberOfRealFiles]
    dataLabels = [
        f"Fake Images\n{numberOfFakeFiles}",
        f"Real Images\n{numberOfRealFiles}",
    ]

    fig, graph = plt.subplots()

    explode = (0, 0.1)

    plt.pie(
        dataSize,
        explode=explode,
        labels=dataLabels,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )

    graph.axis("equal")

    graph.set_title("Database Information")

    plt.show()
    plt.close(fig)

def generate_variations(image_path):
    import os
    import cv2
    import numpy as np

    base_dir = os.path.dirname(os.path.abspath(__file__))
    fake_dir = os.path.join(base_dir, "generated", "fake")
    real_dir = os.path.join(base_dir, "generated", "real")

    os.makedirs(fake_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))

    # 🔹 Generate REAL-like images (light changes)
    for i in range(10):
        img = image.copy()

        # Slight brightness change
        alpha = 1 + np.random.uniform(-0.1, 0.1)
        beta = np.random.randint(-10, 10)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Small rotation
        angle = np.random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((128, 128), angle, 1)
        img = cv2.warpAffine(img, M, (256, 256))

        cv2.imwrite(os.path.join(real_dir, f"real_{i}.jpg"), img)

    # 🔹 Generate FAKE-like images (manipulations)
    for i in range(10):
        img = image.copy()

        # Add noise
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

        # Blur
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Compression artifact simulation
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(10, 40)]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encimg, 1)

        cv2.imwrite(os.path.join(fake_dir, f"fake_{i}.jpg"), img)

if __name__ == "__main__":

    model = joblib.load("model.pkl")

    global width, height
    mainwindow = tk.Tk()

    mainwindow.title("Histogram-Based Fake Image Detection and Performance Analysis")

    label = tk.Label(
        mainwindow,
        text="Histogram-Based Fake Image Detection and Performance Analysis",
        font=("Arial", 25),
    )
    label.place(x=340, y=100)

    width = mainwindow.winfo_screenwidth()
    height = mainwindow.winfo_screenheight()

    mainwindow.geometry("%dx%d" % (width, height))

    mainwindow.minsize(width, height)
    mainwindow.maxsize(width, height)

    upload = tk.Button(
        mainwindow, text="UPLOAD", width=20, height=5, command=imageUpload
    )
    upload.place(x=635, y=400)

    actionButton = tk.Button(
        mainwindow, text="DETECT", width=20, height=5, command=imageDetect
    )

    uploadedImage = tk.Label(mainwindow)
    uploadedImage.place(x=100, y=50)

    result_label = tk.Label(mainwindow, text="", font=("Arial", 24))
    result_label.place(x=650, y=250)

    analysis_label = tk.Label(mainwindow, text="", font=("Arial", 18))
    analysis_label.place(x=650, y=200)

    analysis = tk.Label(mainwindow, text="ANALYSIS", font=("Arial", 30))

    graph_frame = tk.Frame(mainwindow)
    graph_frame.place(x=950, y=0)

    database_info = tk.Button(
        mainwindow,
        text="Database Info",
        width=20,
        height=5,
        command=database_information,
    )
    database_info.place(x=635, y=500)

    generate_btn = tk.Button(mainwindow, text="Generate Variations", command=lambda: generate_variations(imagePath),width=20,height=5)
    

    mainwindow.mainloop()
