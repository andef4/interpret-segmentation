from PIL import Image, ImageDraw
import random
from pathlib import Path
import shutil
import numpy as np


def random_scale():
    return 1.2 - (random.random() * 0.4)

def rect(draw, x, y, w, h, fill=255):
    draw.rectangle((x, y, x + w, y + h), fill=fill)

def circle(draw, x, y, fill=255):
    size = 50 * random_scale()
    draw.ellipse((x, y, x + size, y + size), fill=fill)

def cross(draw, x, y, fill=255):
    scale = random_scale()
    rect(draw, x, y, 10 * scale, 40 * scale, fill)
    rect(draw, x - 15* scale, y + 15 * scale, 40 * scale, 10 * scale, fill)

def triangle(draw, x, y, fill=255):
    scale = random_scale()
    draw.polygon([(x, y), (x + 50*scale, y), (x + 25*scale, y + 50*scale)], fill=fill)

def generate_image(path):
    image = Image.new('L', (240, 240))
    draw = ImageDraw.Draw(image)

    c = lambda: cross(draw, 40 + random.randint(0, 20), 20 + random.randint(0, 20), random.randint(100, 255))
    t = lambda: triangle(draw, 20 + random.randint(0, 20), 110 + random.randint(0, 20), random.randint(100, 255))

    if random.randint(0, 1) == 0:
        c()
        array = np.array(image)
        array = (array > 0).astype(np.uint8) * 255
        segment_image = Image.fromarray(array)
        segment_image.save(path / 'segment.png')
        t()
        scale = random_scale()
        rect(
            draw,
            120 + random.randint(0, 20),
            70 + random.randint(0, 20),
            30*scale,
            30*scale,
            fill=random.randint(100, 255)
        )
    else:
        t()
        image.save(path / 'segment.png')
        c()
        circle(
            draw,
            120 + random.randint(0, 20),
            70 + random.randint(0, 20),
            fill=random.randint(100, 255)
        )

    image.save(path / 'image.png')

if __name__ == '__main__':
    OUTPUT = Path('dataset')
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir()
    for i in range(1000):
        image_folder = OUTPUT / str(i)
        image_folder.mkdir()
        generate_image(image_folder)
