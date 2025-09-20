import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class OrigamiPiece:
    def __init__(self, associatedGraphEle, centroidPos=None, width=None, height=None):
        """
        associatedGraphEle : anything (e.g. a node ID or an edge tuple)
            The graph element this piece is linked to.  (Required.)
        centroidPos : tuple of float (x, y)
            Center position of the rectangle.
        width : float
            Width of the rectangle.
        height : float
            Height of the rectangle.
        """
        self.associatedGraphEle = associatedGraphEle
        self.centroidPos = centroidPos
        self.width = width
        self.height = height

    def draw(self, ax, **rect_kwargs):
        """
        Draws a rectangle on the given matplotlib Axes `ax`, centered at
        `self.centroidPos` with dimensions (`self.width`, `self.height`).

        Additional keyword args are passed to the Rectangle artist
        (e.g. edgecolor='black', facecolor='none').
        """
        if None in (self.centroidPos, self.width, self.height):
            raise ValueError(
                "centroidPos, width and height must all be set before calling draw()"
            )

        cx, cy = self.centroidPos
        # lower-left corner:
        x0 = cx - self.width / 2
        y0 = cy - self.height / 2

        rect = Rectangle((x0, y0),
                         self.width,
                         self.height,
                         **rect_kwargs)
        ax.add_patch(rect)
        return rect