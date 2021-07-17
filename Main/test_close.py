import cv2

cap = cv2.VideoCapture(r"..\sources\a.mp4")

for i in range(100):
    ret, frame = cap.read()
    cv2.imwrite(f"..\\Sources\\A\\{str(i)}.png", frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
