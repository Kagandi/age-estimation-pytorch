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

try:
    import face_recognition_models
except Exception:
    print("Please install `face_recognition_models` with this command before using `face_recognition`:\n")
    print("pip install git+https://github.com/ageitgey/face_recognition_models")
    quit()


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


def yield_images_from_dir(img_dir):
    img_dir = Path(img_dir)

    for img_path in img_dir.glob("*.*"):
        try:
            yield face_recognition.load_image_file(img_path), img_path.name
        except:
            pass


def get_age(output_dir, img_dir, resume_path=None):
    cfg.freeze()

    if output_dir is not None:
        if img_dir is None:
            raise ValueError("=> --img_dir argument is required if --output_dir is used")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model(model_name=cfg.MODEL.ARCH, pretrained=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if resume_path is None:
        resume_path = Path(__file__).resolve().parent.joinpath("misc", "epoch044_0.02343_3.9984.pth")

    if not resume_path.is_file():
        print(f"=> model path is not set; start downloading trained model to {resume_path}")
        url = "https://github.com/yu4u/age-estimation-pytorch/releases/download/v1.0/epoch044_0.02343_3.9984.pth"
        urllib.request.urlretrieve(url, str(resume_path))
        print("=> download finished")

    if Path(resume_path).is_file():
        # print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        # print("=> loaded checkpoint '{}'".format(resume_path))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))

    if device == "cuda":
        cudnn.benchmark = True

    model.eval()
    margin = args.margin
    img_dir = img_dir

    cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
    detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

    img_size = cfg.MODEL.IMG_SIZE
    image_generator = yield_images_from_dir(img_dir)

    with torch.no_grad():
        for img, name in image_generator:
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = np.shape(input_img)

            # detect faces using dlib detector
            detected = [r.rect for r in detector(input_img, 1)]

            faces = np.empty((len(detected), img_size, img_size, 3))

            if len(detected) > 0:
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - margin * w), 0)
                    yw1 = max(int(y1 - margin * h), 0)
                    xw2 = min(int(x2 + margin * w), img_w - 1)
                    yw2 = min(int(y2 + margin * h), img_h - 1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                    faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))

                # predict ages
                inputs = torch.from_numpy(np.transpose(faces.astype(np.float32), (0, 3, 1, 2))).to(device)
                outputs = F.softmax(model(inputs), dim=-1).cpu().numpy()
                ages = np.arange(0, 101)
                predicted_ages = (outputs * ages).sum(axis=-1)

                return zip(faces, predicted_ages)
