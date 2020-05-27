from pathlib import Path
import urllib.request
import numpy as np
import cv2
import dlib
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from model import get_model
from defaults import _C as cfg
import face_recognition
from face_recognition.api import _rect_to_css, _trim_css_to_bounds


try:
    import face_recognition_models
except Exception:
    print("Please install `face_recognition_models` with this command before using `face_recognition`:\n")
    print("pip install git+https://github.com/ageitgey/face_recognition_models")
    quit()


class AgeEstimator(object):

    def __init__(self, model_path=None, margin=0.4):
        cfg.freeze()
        self.model = get_model(model_name=cfg.MODEL.ARCH, pretrained=None)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)

        model_path = Path(model_path)

        if model_path is None:
            model_path = Path(__file__).resolve().parent.joinpath("misc", "epoch044_0.02343_3.9984.pth")

        if not model_path.is_file():
            print(f"=> model path is not set; start downloading trained model to {model_path}")
            url = "https://github.com/yu4u/age-estimation-pytorch/releases/download/v1.0/epoch044_0.02343_3.9984.pth"
            urllib.request.urlretrieve(url, str(model_path))
            print("=> download finished")

        if model_path.is_file():
            # print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(checkpoint['state_dict'])
            # print("=> loaded checkpoint '{}'".format(model_path))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(model_path))

        if device == "cuda":
            cudnn.benchmark = True

        self.model.eval()

    def get_age(self, img):

        cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
        detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

        img_size = cfg.MODEL.IMG_SIZE

        with torch.no_grad():
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = np.shape(input_img)

            # detect faces using dlib detector
            detected = [r.rect for r in detector(input_img, 1)]

            faces = np.empty((len(detected), img_size, img_size, 3))

            if len(detected) > 0:
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - self.margin * w), 0)
                    yw1 = max(int(y1 - self.margin * h), 0)
                    xw2 = min(int(x2 + self.margin * w), img_w - 1)
                    yw2 = min(int(y2 + self.margin * h), img_h - 1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                    faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))

                # predict ages
                inputs = torch.from_numpy(np.transpose(faces.astype(np.float32), (0, 3, 1, 2))).to(device)
                outputs = F.softmax(self.model(inputs), dim=-1).cpu().numpy()
                ages = np.arange(0, 101)
                predicted_ages = (outputs * ages).sum(axis=-1)
                return zip([_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in detected], predicted_ages)
