# Big Data Project II – Emotion Classification with CNN

This project is about classifying emotions from text using a CNN-based model.  
You can download the dataset and run the experiments as described below.

---

## Dataset

Download the dataset using the following link:  
🔗 [Emotion Dataset (Google Drive)](https://drive.google.com/file/d/19QW_oHzMBhGOqbCIirahrduUgc4BxjRX/view?usp=drive_link)

> Please unzip the file in the **same directory** as `main_cnn.py`.

---

## Environment Setup

All experiments were conducted inside a **Docker container** on a **Linux environment**.

### 1. Run Docker Container

```bash
docker run -itd \
  -v /bigdata_project2/:/workspace/bigdata_project2 \
  --ipc=host \
  --name bigdata \
  tensorflow/tensorflow:latest-gpu
```

### 2. Access the Container

```bash
docker exec -it bigdata /bin/bash
cd workspace/bigdata_project2/bigdata
```

---

## Dependencies

Before running the code, install the required Python libraries:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

---

## Run the Model

Once everything is set up, run the CNN-based emotion classifier using:

```bash
python main_cnn.py
```

---

Feel free to modify the code or dataset for your own experimentation!
