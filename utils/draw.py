# 绘图相关
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics.utils.plotting import Annotator, colors

        
def draw_box(
        im0,
        det,
        names: list,
        line_width=4,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        example="abc"
        ):
    if len(det):
        im = im0.copy()
        annotator = Annotator(im, line_width=line_width, font_size=font_size,
                              font=font, pil=pil, example=example)
        if len(det[0]) == 6:  # 目标检测
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f"{names[c]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(c, True))
        elif len(det[0]) == 7:  # 目标跟踪
            for *xyxy, id, conf, cls in reversed(det):
                c = int(cls)
                id = int(id)
                label = f"{id} {names[c]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(c, True))
        annotated_frame = annotator.result()
        return annotated_frame
    else:
        return im0


def center_draw_chinese(xy, im, text, txt_color=(255, 255, 255), font_size=None, font='./font/simkai.ttf', anchor="nw"):
    """
    anchor: str, 文本锚点，可选值有：'nw', 'ne', 'center', 'sw', 'se'
    """
    input_is_pil = isinstance(im, Image.Image)
    im = im if input_is_pil else Image.fromarray(im)
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(im)
    # 字体的格式，"simsun.ttc"  simkai.ttf
    size = font_size or max(round(sum(im.size) / 2 * 0.035), 12)
    fontStyle = ImageFont.truetype(str(font), size)
    # 绘制文本
    w, h = fontStyle.getsize(text)  # text width, height
    if anchor == "nw":
        location = (xy[0], xy[1])
    elif anchor == "ne":
        location = (xy[0] - w, xy[1])
    elif anchor == "center":
        location = (xy[0] - w / 2, xy[1] - h / 2)
    elif anchor == "sw":
        location = (xy[0], xy[1] - h)
    elif anchor == "se":
        location = (xy[0] - w, xy[1] - h)
    draw.text(location, text, txt_color, font=fontStyle)
    return np.asarray(im)


def create_void_img(shape=(1920, 1080), text='no video'):
    """
    创建一个黑色图像，用于显示无视频时的画面
    """
    void_img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
    # void_img = cv2.putText(void_img, text, (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    void_img = center_draw_chinese((int(shape[0]/2), int(shape[1]/2)), void_img, text, anchor="center")
    return void_img