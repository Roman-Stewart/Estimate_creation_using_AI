# Estimate_creation_using_AI
Create a estimate using OCR model
## OCR model generator
 OCR model generator에서 학습데이터를 만들 때 사용한 데이터셋은 AI-Hub의 대용량 손글씨 OCR 데이터 (https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=realm&dataSetSn=605) 이다. 이는 한글 손글씨 데이터가 입력되어 있는 데이터셋이다. 한글 손글씨 이미지 파일과 단어별 라벨 데이터들이 있는 json 파일이 1대1 대응으로 구성돼있다. 

 1. json 파일에서 30번째 줄부터 있는 라벨 데이터, 바운딩 박스 좌표들을 추출해 가져온다.
 2. 이미지를 로드하고 이미지를 바운딩 박스 좌표대로 자르고 전처리해 라벨 데이터와 대응하도록 x와 y 변수에 넣는다.
 3. y 변수에 있는 라벨 데이터는 라벨 인코딩과 원 핫 인코을 해준다.
 4. 이미지를 변형하면서 학습데이터를 생성해 데이터 갯수를 늘린다.
 5. 은닉층을 여러개 만들어 모델을 생성한다. optimizer(learning rate), loss, metrics를 지정한다.
 6. x와 y에 있는 데이터를 이용해 모델 학습을 실시한다.
 7. 모델과 라벨 인코더를 저장한다.
 8. 새로운 이미지에서 예측해본다.
