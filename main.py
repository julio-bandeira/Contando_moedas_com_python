import cv2
import numpy as np
from keras.models import load_model

class Camera:
    def __init__(self):
        # Define os valores iniciais
        self.Kernel_Blur = 5
        self.SigmaX = 3
        self.Threshold_1 = 90
        self.Threshold_2 = 140
        self.Kernel = 2
        self.Dilate_Iterations = 2
        self.Erode_Iterations = 1
        self.Min_Area = 600
        self.video = cv2.VideoCapture(4,cv2.CAP_DSHOW)
        self.model = load_model('Keras_model.h5',compile=False)
        self.data = np.ndarray(shape=(1,224,224,3),dtype=np.float32)
        self.classes = ["1 real","25 cent","50 cent"]

    def preProcess(self, img):
        imgPre = cv2.GaussianBlur(img,(self.Kernel_Blur,self.Kernel_Blur),self.SigmaX)
        imgPre = cv2.Canny(imgPre,self.Threshold_1,self.Threshold_2)
        kernel = np.ones((self.Kernel,self.Kernel),np.uint8)
        imgPre = cv2.dilate(imgPre,kernel,iterations=self.Dilate_Iterations)
        imgPre = cv2.erode(imgPre,kernel,iterations=self.Erode_Iterations)
        return imgPre

    def updateValues_Kernel_Blur(self, pos): self.Kernel_Blur = int(1 + (2*pos))
    def updateValues_SigmaX(self, pos): self.SigmaX = int(pos)
    def updateValues_Threshold_1(self, pos): self.Threshold_1 = int(pos)
    def updateValues_Threshold_2(self, pos): self.Threshold_2 = int(pos)
    def updateValues_Kernel(self, pos): self.Kernel = int(pos)
    def updateValues_Dilate_Iterations(self, pos): self.Dilate_Iterations = int(pos)
    def updateValues_Erode_Iterations(self, pos): self.Erode_Iterations = int(pos)
    def updateValues_Min_Area(self, pos): self.Min_Area = int(pos)

    def callPainel(self):
        # Cria a janela com o controle deslizante
        cv2.namedWindow("Painel")
        cv2.resizeWindow("Painel", 480, 480)
        cv2.createTrackbar("Kernel Blur", "Painel", self.Kernel_Blur, 10, self.updateValues_Kernel_Blur)
        cv2.createTrackbar("SigmaX", "Painel", self.SigmaX, 10, self.updateValues_SigmaX)
        cv2.createTrackbar("Threshold 1", "Painel", self.Threshold_1, 200, self.updateValues_Threshold_1)
        cv2.createTrackbar("Threshold 2", "Painel", self.Threshold_2, 200, self.updateValues_Threshold_2)
        cv2.createTrackbar("Kernel", "Painel", self.Kernel, 10, self.updateValues_Kernel)
        cv2.createTrackbar("Dilate Iterations", "Painel", self.Dilate_Iterations, 10, self.updateValues_Dilate_Iterations)
        cv2.createTrackbar("Erode Iterations", "Painel", self.Erode_Iterations, 10, self.updateValues_Erode_Iterations)
        cv2.createTrackbar("Min. Area", "Painel", self.Min_Area, 2000, self.updateValues_Min_Area)

    def DetectarMoeda(self, img):
        imgMoeda = cv2.resize(img,(224,224))
        imgMoeda = np.asarray(imgMoeda)
        imgMoedaNormalize = (imgMoeda.astype(np.float32)/127.0)-1
        self.data[0] = imgMoedaNormalize
        prediction = self.model.predict(self.data)
        index = np.argmax(prediction)
        percent = prediction[0][index]
        classe = self.classes[index]
        return classe,percent

    def loopCamera(self):
        self.callPainel()
        while True:
            _,img = self.video.read()
            img = cv2.resize(img,(640,480))
            imgPre = self.preProcess(img)

            countors,hi = cv2.findContours(imgPre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


            qtd = 0
            for cnt in countors:
                area = cv2.contourArea(cnt)
                if area > self.Min_Area:
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                    recorte = img[y:y +h,x:x+ w]
                    classe, conf = self.DetectarMoeda(recorte)
                    if conf >0.7:
                        cv2.putText(img,str(classe),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
                        if classe == '1 real': qtd+=1
                        if classe == '25 cent': qtd += 0.25
                        if classe == '50 cent': qtd += 0.5
        
            cv2.rectangle(img,(430,30),(600,80),(0,0,255),-1)
            cv2.putText(img,f'R$ {qtd}',(440,67),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)

            cv2.imshow('Imagem Original',img)
            cv2.imshow('Imagem Processada',imgPre)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

camera = Camera()
camera.loopCamera()