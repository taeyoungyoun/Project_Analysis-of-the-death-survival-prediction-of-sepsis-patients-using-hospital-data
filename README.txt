MobaXterm 
jupyter-lab에서 작업

1. 제출파일 목록
    1) 0.install.ipynb : 필요한 라이브러리를 설치하는 코드입니다.
    2) 1.data_processing.ipynb : 원본 MIMIC-III 데이터에서 data.csv와 label.csv를 생성하는 코드입니다.
    3) 2.mimic_pretrain_transfer_withData.ipynb : data.csv로 모델의 사전학습을 진행하고, data.csv와 label.csv를 활용하여 본 학습을 진행하여 환자의 생/사를 예측합니다. 
    4) data.csv : 패혈증 환자의 모든 처방, 검사결과, 시술이 담겨있는 파일입니다. 
    5) label.csv : 패혈증 환자의 퇴원 시각과 생사여부가 저장되어있는 파일입니다.
    6) REAME.txt : 코드 실행 방법 설명


2. Data 획득 방법
    MIMIC-III 데이터를 csv 형식으로 변환 후 사용합니다. MIMIC-III 데이터에 액세스하려면 의료 연구자 또는 연구 기관에서의 사용 승인이 필요하며, 승인에는 2~3일 정도 소요됩니다. 승인 절차를 거친 뒤에 PhysioNet에서 접근할 수 있습니다.  


3. 실행 필요 프로그램과 라이브러리
    1) python
    2) jupyter lab
    3) 코드 실행에 필요한 파이썬 라이브러리
        -numpy
        -pandas
        -datetime
        -tqdm
        -pytorch
        -sklearn
    4) CUDA
                
4. 실행방법
    case 1. MIMIC-III 데이터를 다운로드 받아 실행시키는 법
    (1) 제출한 압축 파일의 압축을 풉니다.
    (2) 코드 파일(*.ipynb)들이 동일한 작업 폴더 안에 위치하는지 확인합니다. (data.csv, label.csv 불필요)
    (3) csv 파일로 저장한 MIMIC-III 원본 데이터 중 ADMISSIONS.csv, PATIENTS.csv, LABEVENTS.csv, PROCEDUREEVENTS_MV.csv, PRESCRIPTIONS.csv, D_ICD_DIAGNOSES.csv, DIAGNOSES_ICD.csv를 README.txt가 있는 폴더에 같이 넣습니다.
    (4) 0.install.ipynb 코드부터 차례로 실행합니다.
    
    case 2. 제출한 데이터 파일(data.csv, label.csv)를 사용하는 경우
    (1) 제출한 압축 파일의 압축을 풉니다.
    (2) 코드 파일(*.ipynb)들이 동일한 작업 폴더 안에 위치하는지 확인합니다. (data.csv, label.csv 필요)
    (3) 1.dataprocessing.ipynb 코드를 제외하고 0.install.ipynb와 2.mimic_pretrain_transfer_withData를 각각 실행합니다.


5. 학습 환경
    - CPU : i9-13900KF (24 core, 32 threads)
    - RAM : 64GB
    - SSD : 2TB
    - GPU : RTX4090 (VRAM 24GB)


웹페이지 실행
저희 모델은 이미 MIMIC-||| 데이터로 학습을 거친 모델입니다. 여기서 한발 더 나아가, 마이데이터 활용에 동의한 특정 범주의 환자들의 데이터를 입력받아, 추가로 모델을 학습시키는 과정을 구상했습니다. 
특정 병원에서의 환자 데이터를 입력할 수도 있고, 특정 질병 합병증 환자의 데이터를 입력할 수도 있는 등 특정 공통점을 가진 환자의 데이터를 입력해 맞춤형 모델을 생성할 수 있는 것입니다. 
기본 제공 모델 뿐만 아니라 이러한 맞춤형 모델도 생성하고 선택할 수 있다면 좀 더 개인화된 의료 서비스에 가까워질 수 있을 것이라 생각했습니다. 



fastapi와 html을 활용해서 웹 어플리케이션을 구현했습니다. Fastapi는 python 기반이며, 
빠른 개발에 특화된 웹 프레임워크로, 비동기 지원이란 장점이 있어 시간이 오래 걸리는 빅데이터 처리 작업에 적합할 것으로 판단했습니다. 

(1) MobaXterm실행창에서 fastapi폴더까지 이동
(2) uvicorn main12-step5:app --host 0.0.0.0 --port 5000 --reload 입력


- 기본적으로 입력해야 할 데이터는, 환자 1명의 데이터와, 예측받을 날짜 2개
- 추가 학습을 원한다면, 입력할 환자의 data와 생/사 여부 label을 새로운 모델 칸에 각각 업로드

업로드하면 결과 데이터와 날짜 그래프가 나타나게 됩니다. 결과 데이터에서는 날짜와 날짜에 따른 위험도, 그리고 예측된 생/사가 표시되고, 
각 날짜별 위험도를 시각화하여 그래프로 그려줍니다




더 자세한 내용 : https://www.taeyoung-portfolio.com/project/mimic%20%7C%7C%7C%20-%20data



