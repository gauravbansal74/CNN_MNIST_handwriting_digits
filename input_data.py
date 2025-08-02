import cv2
import numpy as np

# global coordinates and drawing state
x = 0
y = 0
drawing  = False

def draw(event, current_x, current_y, flags, param):
    # hook up global variables
    global x, y, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        print("EVENT_LBUTTONDOWN")
        x = current_x
        y = current_y
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            print("EVENT_MOUSEMOVE", drawing)
            cv2.line(canvas, (current_x, current_y), (x, y), (255, 0, 0), 10)
            x, y = current_x, current_y
    elif event == cv2.EVENT_LBUTTONUP:
        print("EVENT_LBUTTONUP")
        drawing = False


# init canvas
canvas = np.zeros((512, 512, 3), np.uint8)
# make canvas background white
canvas.fill(255)
# display the canvas in a window
cv2.imshow('Draw', canvas)
# bind the mouse events
cv2.setMouseCallback('Draw', draw)
while True:
    cv2.imshow('Draw', canvas)
    # break out of a program
    if cv2.waitKey(0) & 0xFF == 27:
        cv2.imwrite('another_sample.png', canvas)
        break

# Clean up windows
cv2.destroyAllWindows()
