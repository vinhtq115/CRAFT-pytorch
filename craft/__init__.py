from craft_detector.run_craft import *
from craft_detector.downloader import *
import os

if not os.path.isfile('craft_mlt_25k.pth'):
    print("downloading model")
    download_file_from_google_drive('1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ', 'craft_mlt_25k.pth')

net = CRAFT()
net.load_state_dict(copyStateDict(torch.load('craft_mlt_25k.pth', map_location='cpu')))
net.eval()
if torch.cuda.is_available():
    net.cuda()
refine_net = None


def detect_text(img, text_threshold=0.7, link_threshold=0.4, low_text=0.4):
    bboxes, polys, score_text = test_net(net, img, text_threshold, link_threshold, low_text, torch.cuda.is_available(), False,
                                         refine_net)
    return bboxes, polys, score_text


def show_bounding_boxes(img, bboxes):
    img = np.array(img)
    for i, box in enumerate(bboxes):
        poly = np.array(box).astype(np.int32).reshape((-1))

        poly = poly.reshape(-1, 2)
        cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)

    return img
