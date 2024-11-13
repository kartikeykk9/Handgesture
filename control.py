import cv2
import mediapipe as mp
import os
import numpy as np
from pynput.keyboard import Controller, Key

kbd = Controller()
gme = r"C:\XboxGames\Asphalt Legends Unite\Content\Asphalt9_gdk_x64_rtl.exe"

def gme_ctrl():
    try:
        os.startfile(gme)
        print("Game launched successfully.")
    except FileNotFoundError:
        print("Game path not found.")
        return
    
    drw = mp.solutions.drawing_utils
    hnd = mp.solutions.hands
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    with hnd.Hands(
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hds:
        
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                continue

            img = cv2.GaussianBlur(img, (5, 5), 0)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lwr = np.array([0, 20, 70], dtype=np.uint8)
            upr = np.array([20, 255, 255], dtype=np.uint8)
            msk = cv2.inRange(hsv, lwr, upr)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = hds.process(rgb)
            hgt, wdt, _ = img.shape

            pos = []
            txt = "Status: Idle"

            if res.multi_hand_landmarks:
                for lmks in res.multi_hand_landmarks:
                    drw.draw_landmarks(img, lmks, hnd.HAND_CONNECTIONS)

                    x_crd = [int(lmk.x * wdt) for lmk in lmks.landmark]
                    y_crd = [int(lmk.y * hgt) for lmk in lmks.landmark]
                    x_min, x_max = max(min(x_crd) - 20, 0), min(max(x_crd) + 20, wdt)
                    y_min, y_max = max(min(y_crd) - 20, 0), min(max(y_crd) + 20, hgt)

                    hnd_msk = msk[y_min:y_max, x_min:x_max]
                    cnts, _ = cv2.findContours(hnd_msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img[y_min:y_max, x_min:x_max], cnts, -1, (0, 255, 0), 2)

                    wrs = lmks.landmark[hnd.HandLandmark.WRIST]
                    pix = drw._normalized_to_pixel_coordinates(wrs.x, wrs.y, wdt, hgt)
                    if pix:
                        pos.append(list(pix))

            if len(pos) == 2:
                l_y = pos[0][1]
                r_y = pos[1][1]
                thr = int(3 * hgt / 4)

                if l_y > thr and r_y > thr:
                    print("Nitro detected.")
                    kbd.press(Key.space)
                    txt = "Status: Accelerating"
                else:
                    kbd.release(Key.space)
                    txt = "Status: Normal"

                if r_y < l_y - 30:
                    print("Turn right detected.")
                    kbd.release('a')
                    kbd.press('d')
                    txt = "Status: Right Turn"

                elif l_y < r_y - 30:
                    print("Turn left detected.")
                    kbd.release('d')
                    kbd.press('a')
                    txt = "Status: Left Turn"

                else:
                    print("Keeping straight.")
                    kbd.release('a')
                    kbd.release('d')
                    kbd.press('w')
                    kbd.release('s')
                    txt = "Status: Moving Forward"

            elif len(pos) == 1:
                print("Backing up detected.")
                kbd.release('a')
                kbd.release('d')
                kbd.release('w')
                kbd.press('s')
                txt = "Status: Backing Up"

            elif len(pos) == 0:
                kbd.release('a')
                kbd.release('d')
                kbd.release('w')
                kbd.release('s')
                kbd.release(Key.space)
                txt = "Status: Idle"

            cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Hand Detector", img)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    gme_ctrl()
