import cv2
import numpy as np
import os


rm = "090174"
rm_sum = sum(int(d) for d in rm)
while rm_sum >= 10:
    rm_sum = sum(int(d) for d in str(rm_sum))
video_file = "q1A.mp4" if rm_sum <= 5 else "q1B.mp4"

cap = cv2.VideoCapture(f"q1/{video_file}")

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()
else:
    print("Vídeo aberto com sucesso!")

frame_count = 0
os.makedirs('output_frames', exist_ok=True)

collision_happened = False
ultrapassou = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_cnt = None
    red_rect = None
    blue_rect = None

    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)
            if area > max_area:
                max_area = area
                max_cnt = cnt
            red_rect = cv2.boundingRect(cnt)

    
    for cnt in contours_blue:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(frame, [cnt], -1, (255, 0, 0), 2)
            if area > max_area:
                max_area = area
                max_cnt = cnt
            blue_rect = cv2.boundingRect(cnt)

    if max_cnt is not None:
        x, y, w, h = cv2.boundingRect(max_cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, "Maior Massa", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if red_rect and blue_rect:
        rx, ry, rw, rh = red_rect
        bx, by, bw, bh = blue_rect
        rect_r = (rx, ry, rx + rw, ry + rh)
        rect_b = (bx, by, bx + bw, by + bh)

        if (rect_r[0] < rect_b[2] and rect_r[2] > rect_b[0] and
            rect_r[1] < rect_b[3] and rect_r[3] > rect_b[1]):
            collision_happened = True
            cv2.putText(frame, "COLISÃO DETECTADA", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        if collision_happened:
            
            if rect_r[0] > rect_b[2] or rect_b[0] > rect_r[2]:
                ultrapassou = True

    if ultrapassou:
        cv2.putText(frame, "ULTRAPASSAGEM DETECTADA", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    frame_filename = f"output_frames/frame_{frame_count:04d}.jpg"
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

cap.release()
print(f"Processo finalizado. Frames salvos na pasta 'output_frames/'.")
