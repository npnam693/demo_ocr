from PIL import Image, ImageDraw, ImageFont
from layoutlm_inference import run_inference  as run_moi
from infer_no_preprocess import run_inference as run_cu
import cv2

class colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    END = '\033[0m'  # Reset color

a = {}
fail = []
for i in range(2):
    filepath = "./img/" + (3-len(str(i))) * "0" + str(i) + ".jpg"
    try:
        p1, image1 = run_moi("filepath")
        a[str(i)] = p1
        print('complete', i)

    except Exception as e:
        # Xử lý ngoại lệ ở đây
        # e chứa thông tin về ngoại lệ
        fail.append(i)
        print("fail", i)

print(a)
