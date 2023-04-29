from TkCanvas.canvas import Canvas
from models import Classifier
from preprocessing import preprocess


class DrawingCanvas(Canvas):
    def __init__(self, width=1000, height=1000):
        super().__init__(width, height)
        self.points = []
        self.register_mouse_press(self.on_click)
        self.register_mouse_move(self.on_move)
        self.register_mouse_release(self.on_release)
        self.classifer = Classifier("models/proto.h5")

    def on_click(self, _):
        self.reset()
        self.points.clear()

    def on_move(self, e):
        self.reset()
        self.points.append((e.x, e.y))
        self.stroke_color = "black"
        self.curve(self.points)

    def on_release(self, _):
        img, _ = self.capture(margin=0.1)
        pimg = preprocess(img)
        txt = self.classifer.classify(pimg)
        print(txt)


canvas = DrawingCanvas()
