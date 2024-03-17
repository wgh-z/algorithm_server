# 绘图相关
from typing import Any
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
