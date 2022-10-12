import cv2 

img = cv2.imread('DATA/00-puppy.img')

while True:
    cv2.imshow('PUPPY',img)
    
    #If We have waited 1ms & Pressed the Esc
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
cv2.destroyAllWindows()