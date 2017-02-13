from itertools import combinations, product
from PIL import Image
from os.path import join, abspath, dirname, exists
from os import mkdir
import math
from array import array
import struct


THIS_DIR = dirname(abspath(__file__))
NODE_LEVELS_DIR = join(THIS_DIR, "levels")
MAX_WIDTH = 4096

def powerset(elements):
    return [combinations(elements, k) for k in xrange(0, len(elements)+1)]

def generate_levels(num_nodes):
    nodes = range(1, num_nodes)
    family = powerset(nodes)
    nodes = set(nodes)

    all_levels = []
    for level_num, subset in enumerate(family):
        levels = []
        for go_through in subset:
            end_at = nodes - set(go_through)
            if not end_at:
                break

            level = product(end_at, [go_through])
            levels.extend(level)

        final_level = [0, list(range(1, level_num+1))]
        levels.append(final_level)

        all_levels.append(levels)

    return all_levels


def pack_pixel(data, level=None):
    end_at = data[0]
    go_through = data[1]

    agg_idx = 1

    # rgb
    agg = [end_at, 0, 0]

    for num in go_through:
        agg_idx = int(math.floor(num / 8)) + 1
        val = 1 << (num % 8)
        agg[agg_idx] |= val

    # alpha, which has to be 1.0, because stupid browsers unpack images as
    # pre-multiplied alpha, which ruins our pixels
    alpha = 255
    if level is not None:
        alpha = level

    agg.append(alpha)

    return agg


def generate_padding(amt):
    padding = [0, 0, 0, 0] * amt
    return padding

def write_level(out, data):
    width = len(data) / 4
    height = 1
    padding = 0

    if width > MAX_WIDTH:
        height = int(math.ceil(width / float(MAX_WIDTH)))
        padding = MAX_WIDTH % width
        width = MAX_WIDTH

    if padding:
        data.extend(generate_padding(padding))

    mode = "RGBA"
    img = Image.frombuffer(mode, (width, height), data, "raw", mode, 0, 1)
    img.save(out)


def generate_cpu_levels(num_nodes, level_dir):
    levels = generate_levels(num_nodes)

    node_dir = join(level_dir, str(num_nodes))
    if not exists(node_dir):
        mkdir(node_dir)

    for i, level in enumerate(levels):
        img_file = join(node_dir, str(i) + ".png")
        arr = array("B")


        for entry in level:
            pixel = pack_pixel(entry)
            arr.extend(pixel)

        with open(img_file, "wb") as h:
            write_level(h, arr)


def generate_one_image_levels(arr, num_nodes, num_elements, bitdepth,
        levels_dir):

    num_pixels = len(arr) / num_elements
    width = MAX_WIDTH
    height = int(math.ceil(num_pixels / MAX_WIDTH))

    max_mode = "RGBA"
    mode = max_mode[:num_elements]

    def write(data, out):
        img = Image.frombuffer(mode, (width, height), data, "raw", mode, 0, 1)
        img.save(out)


    levels = generate_levels(num_nodes)
    elements_range = range(0, num_elements)

    max_idx = 0
    for level_num, level in enumerate(levels):
        print level_num, len(level)

        for entry in level:
            pixel = pack_pixel(entry, 255-level_num)
            idx = hash_pixel(pixel) * num_elements

            #if idx/4 in (475136,466944,462848,460800):
                #print level_num,idx

            for i in elements_range:
                arr[idx+i] = pixel[i]

            max_idx = max(max_idx, idx)

    img_file = join(levels_dir, str(num_nodes) + ".png")
    with open(img_file, "wb") as h:
        write(arr, h)


def hash_pixel(pixel):
    r,g,b,a = pixel

    b1 = r << 16
    b2 = g << 8
    b3 = b

    v = b1|b2|b3
    return v



def generate_gpu_levels():
    bitdepth = 8
    num_elements = 4
    # this is not 4 because we can't use the alpha channel
    avail_for_addressing = 3

    # -1 for the end_at byte
    go_through_bytes = avail_for_addressing-1

    # we can only calculate the tour for max_level number of nodes, given that we
    # only have a certain number of bit flags with which to indicate the "go
    # through" nodes
    max_level = go_through_bytes * bitdepth

    # if we were to use every bit available to us for addressing, we would need this
    # many pixels to store all possible addresses
    total_unoptimized_pixels = 2**(avail_for_addressing*bitdepth)

    # we know that since the MSB contains the "end at", which has a maximum value of
    # the max level, the indices contained by the MSB unused bits can be cut off of
    # our array
    used_pixels = 2**(int(math.log(max_level, 2)) + go_through_bytes*bitdepth)

    # now we need to pad it out to be a multiple of MAX_WIDTH
    used_pixels += used_pixels % MAX_WIDTH

    arr = array("B", [0] * num_elements * used_pixels)

    generate_one_image_levels(arr, max_level, num_elements, bitdepth, NODE_LEVELS_DIR)



# for i in xrange(3, 17):
#     print i
#     generate_cpu_levels(i, NODE_LEVELS_DIR)


#generate_gpu_levels()
