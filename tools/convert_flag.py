import cv2
import sys
import os
src = r'E:\gajalakshmi\project\sign_language\img\deaf_flag.jpg'
dst = r'E:\gajalakshmi\project\sign_language\static\img\deaf_flag.png'
print('src=',src)
print('dst=',dst)
img = cv2.imread(src)
if img is None:
    print('ERROR: source image not found:', src)
    sys.exit(2)
# Ensure destination directory exists
os.makedirs(os.path.dirname(dst), exist_ok=True)
ok = cv2.imwrite(dst, img)
print('wrote:', dst, 'ok=', ok)
if not ok:
    sys.exit(3)
