import cv2

#IP Camera Information
scheme = '192.168.1.'
stream = '1'



#'101' is the last octet of the IP for the stream.
host = scheme+'245'
#cap = cv2.VideoCapture('rtsp://'+host+':554/'+stream)

cap = cv2.VideoCapture('rtsp://admin:audreytech*1@192.168.1.245:554')


while True:
    _, frame = cap.read()
    #print(frame)
    frame1 = cv2.resize(frame,(500, 500))
    #print(frame1)
    cv2.imshow(('camera'),frame1)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


    

