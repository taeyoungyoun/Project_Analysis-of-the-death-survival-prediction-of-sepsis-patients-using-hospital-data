from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from starlette.requests import Request
from fastapi.staticfiles import StaticFiles  # 수정: StaticFiles 임포트 추가
import uvicorn
import asyncio
from pydantic import BaseModel
from torch import load as torch_load, Tensor
from datetime import datetime, timedelta
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import Transformer
import numpy as np
import random
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import os
import sys
import numpy
from encoder_prediction_model import EncoderPredictionModel
import importlib
import seaborn as sns

# 날짜 변환 함수
def read_date(line):
    return datetime.strptime(line.split(',')[1], "%Y-%m-%d")

def map_label(value):
    if value == 0:
        return "생존"
    elif value == 1:
        return "사망"
    else:
        return "알 수 없음"

# CSV 파일로부터 텐서 생성 함수
def generate_tensors_from_csv(data_file, label_file, seq_len=10):
    # label_file 파일 읽기
    df = pd.read_csv(label_file, dtype=str)
    # (SUBJECT_ID, DISCHARGE)를 key로 하고, LABEL_DEAD1_ALIVE0를 value로 하는 dictionary 생성
    dic_pd2label = {(row['SUBJECT_ID'], row['DISCHARGE']): row['LABEL_DEAD1_ALIVE0'] for _, row in df.iterrows()}
    
    num_features = None
    lst_Xt = []
    lst_label = []
    lst_patient_date = []

    prev_patient_id = None
    prev_date = None
    buffer = []

    def process_buffer(buffer, num_features):
        if not buffer:  # 버퍼가 비어있는지 확인하기
            return
        # 누락된 날짜 채우기
        full_dates = [buffer[0][1] + timedelta(days=i) for i in range((buffer[-1][1] - buffer[0][1]).days + 1)]
        full_data = []
        data_exist = []
        
        buffer_idx = 0
        for d in full_dates:
            if buffer_idx < len(buffer) and buffer[buffer_idx][1] == d:
                full_data.append(buffer[buffer_idx][2:])
                data_exist.append(True)
                buffer_idx += 1
            else:
                full_data.append([False] * num_features)
                data_exist.append(False)
        
        Xt_slices = []

        for i in range(len(full_data) - seq_len + 1):
            patient = buffer[0][0]
            date = full_dates[i+seq_len-1]
            nextdate = date + timedelta(days=1)
            ratio_Xt_exist = sum(data_exist[i:i+seq_len]) / seq_len
            key = (patient, nextdate.strftime("%Y-%m-%d"))
            if key in dic_pd2label and ratio_Xt_exist >= 0.2:
                lst_patient_date.append([patient, date.strftime("%Y-%m-%d")])
                Xt_slice = torch.tensor(full_data[i:i+seq_len], dtype=torch.bool)

                Xt_slices.append(Xt_slice)
                lst_label.append(bool(int(dic_pd2label[key])))
            
        if Xt_slices != []:
            lst_Xt.extend(torch.stack(Xt_slices, dim=0))

    with open(data_file, 'r') as file:
        header = file.readline()
        num_features = len(header.split(',')) - 2  # 환자ID와 날짜 제외

        for line in file:
            parts = line.strip().split(',')
            patient_id, date = parts[0], read_date(line)

            # 새로운 환자라면, 버퍼 처리 후 초기화
            if prev_patient_id is not None and prev_patient_id != patient_id:
                process_buffer(buffer, num_features)
                buffer = []

            buffer.append([patient_id, date] + [bool(int(float(val))) for val in parts[2:]])
            prev_patient_id = patient_id
            prev_date = date

        # 마지막 버퍼 처리하기
        if prev_patient_id is not None:
            process_buffer(buffer, num_features)

    Xt = torch.stack(lst_Xt, dim=0)
    Yt = torch.tensor(lst_label).view(-1, 1)

    return lst_patient_date, Xt, Yt

# 데이터셋 클래스 정의
class TransferDataset(Dataset):
    def __init__(self, Xt, Yt, type='all', train_size=None, shuffle=False, seed=None):
        self.Xt = Xt
        self.Yt = Yt
        self.type = type
        self.shuffle = shuffle
        self.train_size = train_size

        # 시드 설정 (설정된 경우)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        if self.type == 'all':
            length = len(self.Xt)
        elif self.type == 'train':
            length = int(len(self.Xt) * self.train_size)
        elif self.type == 'test':
            length = int(len(self.Xt) * (1.0 - self.train_size))
        else:
            raise ValueError("Invalid type provided. Expected 'all', 'train', or 'test'.")
        
        self.length = length
        
        # shuffle이 True이면 랜덤 인덱스 생성
        if self.shuffle:
            self.indices = torch.randperm(len(self.Xt))
        else:
            self.indices = torch.arange(len(self.Xt))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.type == 'all':
            real_idx = self.indices[idx]
        elif self.type == 'train':
            real_idx = self.indices[idx]
        elif self.type == 'test':
            real_idx = self.indices[int(len(self.Xt) * self.train_size) + idx]
        return self.Xt[real_idx].float(), self.Yt[real_idx].float()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 정적 파일 디렉토리 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 초기화 및 불러오기
model = EncoderPredictionModel(output_dim=1)
model.load_state_dict(torch.load('transferred_model2.pth', map_location=device))
model.to(device)
model.eval()

# / 엔드포인트를 수정하여 HTMLResponse를 반환하도록 변경
@app.get("/", response_class=HTMLResponse)
async def show_upload_form(request: Request):
    return templates.TemplateResponse("frontblue_v2.html", {"request": request})

@app.post("/show-result/", response_class=HTMLResponse)
async def show_result(request: Request, data_file: UploadFile = File(...), label_file: UploadFile = File(...)):
    # 업로드한 파일을 임시 디렉터리에 저장합니다.
    with open(f"temp_data.csv", "wb") as temp_data_file:
        temp_data_file.write(data_file.file.read())

    with open(f"temp_label.csv", "wb") as temp_label_file:
        temp_label_file.write(label_file.file.read())

    data_file_path = "temp_data.csv"
    label_file_path = "temp_label.csv"

    # 데이터 처리 함수를 호출합니다.
    lst_date, Xt, Yt = generate_tensors_from_csv(data_file_path, label_file_path, seq_len=10)

    os.remove(data_file_path)
    os.remove(label_file_path)

    batch_size = 32
    test_dataset = TransferDataset(Xt, Yt, type='test', train_size=0, shuffle=True, seed=42)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    all_preds = []
    all_true = []

    # 예측
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.permute(1, 0, 2)

            batch_x = batch_x.to(device)

            outputs = model(batch_x)

            all_preds.append(outputs.cpu().numpy())
            all_true.append(batch_y.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    mapped_labels = np.array([map_label(value) for value in all_true], dtype=object)
    all_test = pd.DataFrame({'위험도': np.round(all_preds, 4).flatten(), '생존/사망': mapped_labels.flatten()})
    
    plt.rcParams['font.family'] = 'NanumGothic'
    x_values = range(1, 11)
    y_values = all_test['위험도']
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=x_values, y=y_values, linewidth=2, color='red')
    plt.xticks(x_values)
    plt.xlabel('days')
    plt.ylabel('위\n험\n도', rotation=0, ha='right')
    
    # 그래프 이미지 저장
    plot_image_path = 'static/result_plot.png'
    plt.savefig(plot_image_path)
    plt.close()  # 그래프 창 닫기

    # 결과 데이터와 이미지 경로를 HTML로 렌더링하여 반환합니다.
    return templates.TemplateResponse("result.html", {"request": request, "data": all_test.to_dict(orient='records'), "plot_image": plot_image_path})

@app.get("/favicon.ico")
async def get_favicon():
    return {"message": "No favicon.ico found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5008)
#uvicorn main12:app --host 0.0.0.0 --port 5008 --reload