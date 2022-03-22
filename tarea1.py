from email.policy import default
import cv2
from matplotlib import pyplot as plt
import numpy as np


def main():
    print("Que accion desea realizar?")
    print("1. Deteccion de rasgos")
    print("2. Emparejamiento de rasgos")
    print("3. Salir del programa")
    
    selector = int(input())
    try:                                                                                    #De no ser un numero valido llevara directo a except
        match (selector):                                                                   #Switch 
            case 1:
                detection()
            case 2:
                pairing()
            case 3:
                exit()
            case _:
                print("----- Seleccionar opcion 1 , 2 o 3")
                main()
    except:
        print("----- Seleccionar opcion 1 , 2 o 3")
        main()
   


def detection():                                                                            # CASE 1: Deteccion
    try:                                                                                    #Intenta abrir la imagen dada
        #img = cv2.imread(input("Nombre de la imagen a detectar:"))
        img = cv2.imread('b.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
    except:
        print("----- No se pudo procesar la imagen, intente de nuevo")
        detection()
    
    print("")
    print(" Menu deteccion de rasgos")
    print("1. Good Features to Track")
    print("2. FAST")
    print("3. BRIEF")
    print("4. ORB")
    print("5. AGAST")
    print("6. AKAZE")
    print("7. BRISK")
    print("8. KAZE")
    print("9. SIFT")
    print("10. Regresar menu principal")
    
    detSelector = int(input())
    try:                                                                                    #De no ser un numero valido llevara directo a except
        match (detSelector):                                                                #Switch 
            
            case 1:                                                                         #Good Features to track
                def nothing(val):    
                    pass 
                cv2.namedWindow('Good Features to track')
                cv2.createTrackbar('maxCorners','Good Features to track',0,50, nothing)
                
                while True:
                    
                    val = cv2.getTrackbarPos('maxCorners','Good Features to track')  
                    if val > 0:                                                 
                        corners = cv2.goodFeaturesToTrack(gray.copy(), val, 0.01, 10)
                             
                    else: 
                        corners = cv2.goodFeaturesToTrack(gray.copy(), 1, 0.01, 10)          #Se establece un minimo de uno para no llegar a 0 en maxCorners
                        
                    corners = np.int0(corners)
                    imgN = img.copy()                                                         #Se crea una copia para poder manipular el for, y que este no afecte en la siguiente movimiento del trackbar
                    for i in corners:
                        x, y = i.ravel()
                        cv2.circle(imgN, (x, y), 3, 255, -1)
                        
                    cv2.imshow('Good Features to track',imgN)
                    k = cv2.waitKey(1)
                    if k == 27:
                        break
            
            case 2:                                                                         #FAST
                fastDetector = cv2.FastFeatureDetector_create()                                     #Se inicia el objeto
                kpWith = fastDetector.detect(img, None)                                             #Encuentra y dibuja los puntos de interes
                img2 = cv2.drawKeypoints(img, kpWith, None, color=(255, 0, 0))
                cv2.imshow('FAST with nonmaxSuppression', img2)
                

                fastDetector.setNonmaxSuppression(0)                                                #Desactiva nonmaxSuppression
                kpWithOut = fastDetector.detect(img, None)
                img3 = cv2.drawKeypoints(img, kpWithOut, None, color=(255, 0, 0))
                cv2.imshow('FAST without nonmaxSuppression', img3)
                cv2.waitKey(0)
            
            case 3:                                                                         #BRIEF
                star = cv2.xfeatures2d.StarDetector_create()                                #Se crea detector
                brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()                   #Se crea descriptor
                kp = star.detect(img, None)                                                 
                kp = brief.compute(img, kp)
                img2=cv2.drawKeypoints(img,kp,None,color=(255,0,0))
                cv2.imshow("BRIEF",img2)
                cv2.waitKey(0)

            case 4:                                                                         #ORB
                orbDetector = cv2.ORB_create()                                                      #Se crea el detector                          
                kp = orbDetector.detect(img, None)
                kp, des = orbDetector.compute(img, kp)
                img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
                cv2.imshow("ORB",img2)
                cv2.waitKey(0)
            
            case 5:                                                                         #AGAST
                agastDetector = cv2.AgastFeatureDetector_create()                                   #Se crea el detector 
                kp = agastDetector.detect(img, None)
                img2 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
                cv2.imshow('AGAST with nonmaxSuppression', img2)
                
                agastDetector.setNonmaxSuppression(0)
                kp = agastDetector.detect(img, None)
                img3 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
                cv2.imshow('AGAST without nonmaxSuppression', img3)
                cv2.waitKey(0)

            case 6:                                                                         #AKAZE
                akazeDetector = cv2.AKAZE_create()                                               #Se crea el detector
                kp, des = akazeDetector.detectAndCompute(gray, None)
                img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
                cv2.imshow("AKAZE", img2)
                cv2.waitKey(0)
            
            case 7:                                                                         #BRISK
                briskDetector = cv2.BRISK_create()                                               #Se crea el detector
                kp, des = briskDetector.detectAndCompute(gray, None)
                img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
                cv2.imshow("BRISK", img2)
                cv2.waitKey(0)
            
            case 8:                                                                         #KAZE
                kazeDetector = cv2.KAZE_create()                                                #Se crea el detector
                kp, des = kazeDetector.detectAndCompute(gray, None)
                img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
                cv2.imshow("KAZE", img2)
                cv2.waitKey(0)
                
            case 9:                                                                         #SIFT
                siftDetector = cv2.SIFT_create()                                            #Se crea el detector
                kp = siftDetector.detect(gray, None)
                img2 = cv2.drawKeypoints(gray, kp, img)
                cv2.imshow("SIFT",img2)
                cv2.waitKey(0)    
            case _:
                print("----- Seleccionar opcion entre 0 y 9")
                main()
    except:
        print("----- Seleccionar opcion entre 0 y 9")
        main()

def pairing():                                                                              #CASE 2: Emparejamiento
    try:                                                                                    #Intenta abrir la imagen dada
        img = cv2.imread(input("Nombre de la primera imagen: "),0)
        img2 = cv2.imread(input("Nombre de la segunda imagen: "),0)
        #img = cv2.imread('3.jpg')
        #img2 = cv2.imread('1.jpg')
        
            
    except:
        print("----- No se pudo procesar la alguna imagen, intente de nuevo")
        pairing()
    
    print("")
    print(" Selecciona el metodo para deteccion y descripcion a usar: ")
    print("1. SIFT")
    print("2. KAZE")
    print("3. BRIEF")
    print("4 .BRISK")
    print("5. ORB")
    print("6. AKAZE")
    paiDetSelector=int(input())
    
    print("")
    print(" Selecciona el metodo de emparejamiento: ")
    print("1. FUERZA BRUTA")
    print("2. FLANN")
    paiMetSelector=int(input())
    
    try:                                                                                    #De no ser un numero valido llevara directo a except
        match (paiDetSelector):                                                                #Switch 
                case 1:                                                                     #SIFT                                                                    
                    print("1. SIFT")
                    sift = cv2.SIFT_create()
                    kp, des = sift.detectAndCompute(img, None)
                    kp2, des2 = sift.detectAndCompute(img2, None)
                    try:
                        match (paiMetSelector):
                            case 1:                                                                 #Brute Forze   
                                bf = cv2.BFMatcher()
                                matches = bf.knnMatch(des, des2, k=2)
                                good = []
                                for m, n in matches:
                                    if m.distance < 0.75 * n.distance:
                                        good.append([m])
                                img3 = cv2.drawMatchesKnn(img, kp, img2, kp2, good, None, flags=2)
                                cv2.imshow("SIFT - BRUTE FORZE - MATCHES", img3)
                                cv2.waitKey(0)
                            
                            case 2:                                                                 #FLANN
                                indexPara = dict(algorithm=5, trees=5)
                                findPara = dict(checks=50)  
                                flann = cv2.FlannBasedMatcher(indexPara, findPara)
                                des=np.float32(des)
                                des2=np.float32(des2)
                                matches = flann.knnMatch(des, des2, k=2)
                                matchesMask = [[0,0] for i in range(len(matches))]
                                for i,(m,n) in enumerate(matches):
                                    if m.distance < 0.7 * n.distance:
                                        matchesMask[i] = [1, 0]
                                drawPara = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),matchesMask=matchesMask,flags=0)
                                img3 = cv2.drawMatchesKnn(img, kp, img2, kp2, matches, None, **drawPara)
                                cv2.imshow("SIFT - FLANN - MATCHES", img3)
                                cv2.waitKey(0)
                            
                            case _:
                                print("----- No se pudo realizar")
                    except:
                        print("----- No se pudo realizar SIFT")
                          
                case 2:                                                                     #KAZE
                    print("2. KAZE")
                    kaze = cv2.KAZE_create()
                    kp, des = kaze.detectAndCompute(img, None)
                    kp2, des2 = kaze.detectAndCompute(img2, None)
                    try:
                        match (paiMetSelector):
                            case 1:                                                                 #Brute Forze 
                                bf = cv2.BFMatcher()
                                matches = bf.knnMatch(des, des2, k=2)
                                good = []
                                for m, n in matches:
                                    if m.distance < 0.75 * n.distance:
                                        good.append([m])
                                img3 = cv2.drawMatchesKnn(img, kp, img2, kp2, good, None, flags=2)
                                cv2.imshow("KASE - BRUTE FORZE - MATCHES", img3)
                                cv2.waitKey(0)

                            case 2:                                                                 #FLANN 
                                
                                indexPara = dict(algorithm=1, trees=5)
                                findPara = dict(checks=50)  
                                flann = cv2.FlannBasedMatcher(indexPara, findPara)
                                des = np.float32(des)
                                des2 = np.float32(des2)
                                matches = flann.knnMatch(des, des2, k=2)
                                matchesMask = [[0, 0] for i in range(len(matches))]
                                for i, (m, n) in enumerate(matches):
                                    if m.distance < 0.7 * n.distance:
                                        matchesMask[i] = [1, 0]
                                drawPara = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),matchesMask=matchesMask,flags=0)
                                img3 = cv2.drawMatchesKnn(img, kp, img2, kp2, matches, None, **drawPara)
                                cv2.imshow("KASE - FLANN - MATCHES", img3)
                                cv2.waitKey(0)
                            case _:
                                print("----- No se pudo realizar")
                    
                    except:
                        print("----- No se pudo realizar KAZE")

                case 3:                                                                     #BRIEF
                    print("3. BRIEF")
                    star = cv2.xfeatures2d.StarDetector_create()
                    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
                    kp = star.detect(img, None)
                    kp, des = brief.compute(img, kp)
                    kp2 = star.detect(img2, None)
                    kp2, des2 = brief.compute(img2, kp2)
                    try:
                        match (paiMetSelector):
                            case 1:                                                             #Brute Forze  
                                bf = cv2.BFMatcher()
                                matches = bf.knnMatch(des, des2, k=2)
                                good = []
                                for m, n in matches:
                                    if m.distance < 0.75 * n.distance:
                                        good.append([m])
                                img3 = cv2.drawMatchesKnn(img, kp, img2, kp2, good, None, flags=2)
                                cv2.imshow("BRIEF - BRUTE FORZE - MATCHES", img3)
                                cv2.waitKey(0)    
                            
                            case 2:                                                                 #FLANN 
                                
                                indexPara = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1) 
                                findPara = dict(checks=300)
                                flann = cv2.FlannBasedMatcher(indexPara, findPara)
                                matches = flann.knnMatch(des, des2, k=2)
                                matchesMask = [[0, 0] for i in range(len(matches))]
                                for i, (m, n) in enumerate(matches):
                                    if m.distance < 0.7 * n.distance:
                                        matchesMask[i] = [1, 0]
                                drawPara = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),matchesMask=matchesMask,flags=0)
                                img3 = cv2.drawMatchesKnn(img, kp, img2, kp2, matches, None, **drawPara)
                                cv2.imshow("BRIEF - FLANN - MATCHES", img3)
                                cv2.waitKey(0)

                            case _:
                                print("----- No se pudo realizar")
                    except:
                        print("----- No se pudo realizar BRIEF")          

                case 4:
                    print("4. BRISK")
                    detector = cv2.BRISK_create()
                    kp, des1 = detector.detectAndCompute(img, None)
                    kp2, des2 = detector.detectAndCompute(img2, None)
                    try:
                        match (paiMetSelector):
                            case 1:                                                             #Brute Forze  
                                bf = cv2.BFMatcher()
                                matches = bf.knnMatch(des1, des2, k=2)
                                good = []
                                for m, n in matches:
                                    if m.distance < 0.75 * n.distance:
                                        good.append([m])
                                img3 = cv2.drawMatchesKnn(img, kp, img2, kp2, good, None, flags=2)
                                cv2.imshow("BRISK - BRUTE FORZE - MATCHES", img3)
                                cv2.waitKey(0)                                
                            
                            case 2:                                                             #FLANN 
                                
                                indexPara = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1) 
                                findPara = dict(checks=50)
                                flann = cv2.FlannBasedMatcher(indexPara, findPara)
                                matches = flann.knnMatch(des1, des2, k=2)
                                matchesMask = [[0, 0] for i in range(len(matches))]
                                for i, (m, n) in enumerate(matches):
                                    if m.distance < 0.7 * n.distance:
                                        matchesMask[i] = [1, 0]
                                drawPara = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),matchesMask=matchesMask,flags=0)
                                img3 = cv2.drawMatchesKnn(img, kp, img2, kp2, matches, None, **drawPara)
                                cv2.imshow("BRISK - FLANN - MATCHES", img3)
                                cv2.waitKey(0)

                            case _:
                                print("----- No se pudo realizar")
                    except:
                        print("----- No se pudo realizar BRISK")                    

                case 5:
                    print("5. ORB")
                    orb=cv2.ORB_create()
                    kp, des =orb.detectAndCompute(img,None)
                    kp2, des2 = orb.detectAndCompute(img2,None)
                    try:
                        match (paiMetSelector):
                            case 1:                                                             #Brute Forze  
                                bf = cv2.BFMatcher()
                                matches = bf.knnMatch(des, des2, k=2)
                                good = []
                                for m, n in matches:
                                    if m.distance < 0.75 * n.distance:
                                        good.append([m])
                                img3 = cv2.drawMatchesKnn(img, kp, img2, kp2, good, None, flags=2)
                                cv2.imshow("ORB - matches",img3)
                                cv2.waitKey(0)
                            case 2:                                                             #FLANN 
                                FLANN_INDEX_LSH = 6
                                indexPara = dict(algorithm=5, table_number=6, key_size=12, multi_probe_level=1) 
                                findPara = dict(checks=50)
                                flann = cv2.FlannBasedMatcher(indexPara, findPara)
                                matches = flann.knnMatch(des1, des2, k=2)
                                matchesMask = [[0, 0] for i in range(len(matches))]
                                for i, (m, n) in enumerate(matches):
                                    if m.distance < 0.7 * n.distance:
                                        matchesMask[i] = [1, 0]
                                drawPara = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),matchesMask=matchesMask,flags=0)
                                img3 = cv2.drawMatchesKnn(img, kp, img2, kp2, matches, None, **drawPara)
                                cv2.imshow("ORB - FLANN - MATCHES", img3)
                                cv2.waitKey(0)

                            case _:
                                print("----- No se pudo realizar")
                    except:
                        print("----- No se pudo realizar ORB")                    

                case 6:
                    print("6. AKAZE")
                    akaze=cv2.AKAZE_create()
                    kp, des =akaze.detectAndCompute(img,None)
                    kp2, des2 = akaze.detectAndCompute(img2,None)
                    try:
                        match (paiMetSelector):
                            case 1:                                                             #Brute Forze  
                                bf = cv2.BFMatcher()
                                matches = bf.knnMatch(des, des2, k=2)
                                good = []
                                for m, n in matches:
                                    if m.distance < 0.75 * n.distance:
                                        good.append([m])
                                img3 = cv2.drawMatchesKnn(img, kp, img2, kp2, good, None, flags=2)
                                cv2.imshow("AKAZE - matches",img3)
                                cv2.waitKey(0)
                            case 2:                                                             #FLANN 
                                
                                indexPara = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1) 
                                findPara = dict(checks=50)
                                flann = cv2.FlannBasedMatcher(indexPara, findPara)
                                matches = flann.knnMatch(des, des2, k=2)
                                matchesMask = [[0, 0] for i in range(len(matches))]
                                for i, (m, n) in enumerate(matches):
                                    if m.distance < 0.7 * n.distance:
                                        matchesMask[i] = [1, 0]
                                drawPara = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),matchesMask=matchesMask,flags=0)
                                img3 = cv2.drawMatchesKnn(img, kp, img2, kp2, matches, None, **drawPara)
                                cv2.imshow("AKAZE - FLANN - MATCHES", img3)
                                cv2.waitKey(0)                                
                            case _:
                                print("----- No se pudo realizar")
                    except:
                        print("----- No se pudo realizar AKAZE")                    
                            
    except:
        print("----- Seleccionar opcion valida")
        main()        
             
main()