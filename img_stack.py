################################################################################
# img_stack.py
# Author: Henry Yuan
################################################################################
# 
# Module to stack series of images into a final image in a variety of ways.
#
# Execution command:
# python(3) img_stack.py [out format] [stacking method] [additional args] 
# {[x_min] [x_max] [y_min] [y_max]} < [input filename]
#
# {} denotes optional
#
# Command-line arguments:
#  - output file format (png, jpg)
#  - 0 for create output image (img_stack.[out format]), 1 for write to binary file (img_stack.bin)
#  - 0, 1, or 2: 0 for luminance stacking, 1 for mean stacking, 2 for median
#    stacking
#  - Additional arguments
#       - If luminance: 
#           - 0 for dimmest, 1 for brightest
#           - 0 for brute force, 1 for parallelization
#       - If median: tolerance for rgb color margin (integer between 0 and 255)
#  - (OPTIONAL) x_min, x_max, y_min, y_max,
#    fractions of the total width and height respectively of the image that
#    should be stacked. The pixels that are not stacked are taken 
#    from the first picture in the input.
# 
#
# reads from standard input a list of image file names
# input file should be number of images on first line, then images filenames
# on subsequent lines, 1 per line
# 
# Notes on Runtime: 
# Median:
#   - Takes 238 seconds to stack 5 12-MP photos
#   - should be linear in number of pixels and number of photos
#   - as tolerance increases, runtime decreases
#   - experiment by making tolerance as large as possible that yields
#     desired results.
#
# Mean:
#   - Takes 34 seconds to stack 5 12-MP photos
#   - should be linear in number of pixels and number of photos
#
# Luminance parallelization:
#   - ~15 min for 78 18-MP photos
#
# Notes on Uses:
# Median:
#   - take moving things out of images, like people
#   - good for daytime photos
# 
# Mean:
#   - simulate long-exposure photos with multiple short exposure photos
#     for colorful photos
#   - reduce noise in dark / not very colorful photos, like night photos / 
#     astrophotography
#
# Luminance:
#   - star trails
#
# TO-DO:
#   - check if order of pixel data in memory affects runtime significantly
#   - add ability to do multiple regions, not just 1 region
#
################################################################################

# from PIL import Image
import cv2
import multiprocessing as mp
import sys
import numpy as np
import os.path
import time
import gc
# from numba import jit


# computes the distance between c1 and c2, where c1 and c2 are tuples
# representing colors in rgb space (r,g,b)
def sq_dist(c1, c2):
    return (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2 


# # Returns width and height of images if all images in imgs have the same 
# # width and height. Otherwise returns -1 for both width and height
# def check_dims(imgs):
#     w = imgs[0].width
#     h = imgs[0].height
#     for img in imgs:
#         if img.width != w or img.height != h:
#             return (-1, -1)
#     return (w, h)


# Swaps item at index i with item at index j in arr
def swap(arr, i, j):
    temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp


# Returns median color (r,g,b) of set of colors clrs
def median_clr(clrs, delta):
    # order colors such that the total distance of the path from the
    # first color to the last color in rgb color space is minimized 
    # May not find the actual minimum distance order, but get close
    n = len(clrs)
    sum_r, sum_g, sum_b, = 0, 0, 0
    for c in clrs:
        sum_r += c[0]
        sum_g += c[1]
        sum_b += c[2]

    mean_clr = [sum_r / n, sum_g / n, sum_b / n]

    for i in range(n // 2):
        # if all colors are the same (within error delta), return that color!
        c = clrs[i]
        all_same = True
        for j in range(i + 1, n - i):
            if sq_dist(c, clrs[j]) >= delta:
                all_same = False
                break
    
        if all_same:
            return c

        # recalculate mean if i not 0
        if i > 0:
            for k in range(3):
                mean_clr[k] *= n - (2*(i - 1))
                mean_clr[k] -= (clrs[i-1][k] - clrs[n-i][k])
                mean_clr[k] /= n - (2*i)

        # find color with max distance from this mean
        max_dist = 0
        ind1 = -1
        for j in range(i, n - i):
            d = sq_dist(mean_clr, clrs[j])
            if d > max_dist:
                max_dist = d
                ind1 = j

        # find color with max distance from the previously found color
        max_dist = 0
        ind2 = -1
        for j in range(i, n - i):
            d = sq_dist(clrs[ind1], clrs[j])
            if d > max_dist:
                max_dist = d
                ind2 = j

        # enter these in proper location in list
        if i == 0:
            # order doesn't matter
            swap(clrs, i, ind1)
            swap(clrs, n-i-1, ind2)
        else:
            # find which order minimizes total distance
            if sq_dist(clrs[i-1], clrs[ind1]) <= sq_dist(clrs[i-1], clrs[ind2]):
                swap(clrs, i, ind1)
                swap(clrs, n-i-1, ind2)
            else:
                swap(clrs, i, ind2)
                swap(clrs, n-i-1, ind1)

    return clrs[n // 2]

# helper function for 2 functions below
def lum_stack(imgs, x_range, y_range, brightest):

    start = time.time() ##

    h, w, c = np.shape(imgs[0])
    x_min = int(x_range[0]*w)
    x_max = int(x_range[1]*w)
    y_min = int(y_range[0]*h)
    y_max = int(y_range[1]*h)

    imgf = imgs[0].copy()

    # print('Final image loaded: '+str(time.time()-start)) ##
    # start = time.time() ##

    # calculate luminance of each pixel in each image, find index of desired
    # luminance extremum
    # *** order is BGR not RGB !!! ***
    # bs, gs, rs = np.split(imgs, 3, axis = -1)
    lums = (0.299 * imgs[:,y_min:y_max,x_min:x_max,2]) + (0.587 * imgs[:,y_min:y_max,x_min:x_max,1]) + (0.114 * imgs[:,y_min:y_max,x_min:x_max,0])
    if brightest:
        inds = lums.argmax(axis=0)
    else:
        inds = lums.argmin(axis=0)
    # print('luminances calculated: '+str(time.time()-start)) ##
    # start = time.time() ##

    print('Luminosity maximums found: '+str(time.time()-start)) ##
    start = time.time() ##

    # construct final stacked image
    for i in range(y_min, y_max):
        for j in range(x_min, x_max):
            imgf[i, j] = imgs[inds[i - x_min, j - y_min], i, j]
    
    print('Image constructed: '+str(time.time()-start)) ##
    
    # free memory
    del imgs
    del lums
    del inds
    gc.collect()
    return imgf

# stacks imgs in the fractional range x_range, y_range, returning a final image
# that takes the brightest pixels of each picture if brightest is True,
# dimmest if False.
# Can be used to make star trail photo
# stacks all images at once (brute-force method)
def lum_stack_bf(img_names, x_range, y_range, brightest):
    imgs = np.array([cv2.imread(name.strip(), cv2.IMREAD_UNCHANGED) for name in img_names])
    return lum_stack(imgs, x_range, y_range, brightest)

# Same as above, but uses less memory at a time, by only stacking 2-3
# images at a time and running these in parallel (parallel method)
def lum_stack_par(img_names, x_range, y_range, brightest):
    
    n = len(img_names) # number of total images left
    n_cpu = mp.cpu_count()

    # 2 dictionaries for two levels
    # each dictionary has integer keys and the value is
    # an img if higher level or the img_name if lower level
    low_imgs = {}
    high_imgs = {}
    next_key = 0 # var to keep track of arbitrary key value, just needs to be different for each image
    for fname in img_names:
        low_imgs[next_key] = fname
        next_key += 1

    pool = mp.Pool(n_cpu)
    # each loop should do n_cpu calls to lum_stack_sub()
    while (n > 1):
        three = False
        ready_imgs = []
        while len(ready_imgs) < 2*n_cpu:
            # check high-priority images first
            if len(high_imgs) > 0:
                ready_imgs.append(high_imgs.popitem()[1])
            # then check low priority ones
            elif len(low_imgs) > 0:
                img = cv2.imread(low_imgs.popitem()[1].strip(), cv2.IMREAD_UNCHANGED)
                ready_imgs.append(img)
            # if gets here, no images left
            else:
                break
            
        # ensure EVEN number of images in ready_imgs
        if len(ready_imgs) % 2 != 0:
            # if 3 images left, combine all at once
            if len(ready_imgs) == 3:
                three = True
            else:
                ready_imgs = ready_imgs[:len(ready_imgs)-1]
        
        # run jobs and add resulting images to high_imgs
        if three:
            pool.close()
            return lum_stack(np.array(ready_imgs), x_range, y_range, brightest)
           
        # implicit else
        num_ready = len(ready_imgs)
        args = [(np.array(ready_imgs[2*i:2*i+2]), x_range, y_range, brightest) for i in range(num_ready // 2)]
        results = pool.starmap(lum_stack, args)

        # add results back to dict
        for img in results:
            high_imgs[next_key] = img
            next_key += 1
        n -= len(args)

        # free memory
        del args
        del ready_imgs
        gc.collect()
    
    # close pool
    pool.close()

    # return final image
    if len(high_imgs) > 1:
        print('ERROR')
    return high_imgs.popitem()[1]

    
        
# Stacks imgs by taking mean pixel color for each pixel, returns resulting 
# image. Only stacks pixels in the square defined by x_range and y_range,
# tuples that are fractions of the overall image width and height respectively
# def mean_stack(imgs, x_range, y_range):
#     # check that all pictures have same dimensions and get dimensions
#     w, h = check_dims(imgs)
#     if w == -1:
#         print("Images do not have equal dimensions")
#         exit()

#     n = len(imgs)
#     x_min = int(x_range[0]*w)
#     x_max = int(x_range[1]*w)
#     y_min = int(y_range[0]*h)
#     y_max = int(y_range[1]*h)

#     imgf = imgs[0].copy()
#     pxs = imgf.load()

#     pxArr = [imgs[i].load() for i in range(n)]

#     rs = np.array([[[pxArr[i][j, k][0] for k in range(y_min, y_max)] for j in range(x_min, x_max)] for i in range(n)], dtype=np.uint8)
#     gs = np.array([[[pxArr[i][j, k][1] for k in range(y_min, y_max)] for j in range(x_min, x_max)] for i in range(n)], dtype=np.uint8)
#     bs = np.array([[[pxArr[i][j, k][2] for k in range(y_min, y_max)] for j in range(x_min, x_max)] for i in range(n)], dtype=np.uint8)

#     rs_mean = np.mean(rs, axis=0)
#     gs_mean = np.mean(gs, axis=0)
#     bs_mean = np.mean(bs, axis=0)

#     for i in range(x_min, x_max):
#         for j in range(y_min, y_max):
#             pxs[i, j] = (rs_mean[i - x_min, j - y_min], gs_mean[i - x_min, j - y_min], bs_mean[i - x_min, j - y_min])

#     return imgf


# # stacks imgs by taking median pixel color for each pixel, returns resulting 
# # image. Only stacks pixels in the square defined by x_range and y_range,
# # tuples that are fractions of the overall image width and height respectively
# def median_stack(imgs, x_range, y_range, delta):
#     # check that all pictures have same dimensions and get dimensions
#     w, h = check_dims(imgs)
#     if w == -1:
#         print("Images do not have equal dimensions")
#         exit()

#     n = len(imgs)
#     x_min = int(x_range[0]*w)
#     x_max = int(x_range[1]*w)
#     y_min = int(y_range[0]*h)
#     y_max = int(y_range[1]*h)

#     imgf = imgs[0].copy()
#     pxs = imgf.load()

#     # pxArr = [imgs[i].load() for i in range(n)]

#     # rs = np.array([[[pxArr[i][j, k][0] for k in range(y_min, y_max)] for j in range(x_min, x_max)] for i in range(n)])
#     # gs = np.array([[[pxArr[i][j, k][1] for k in range(y_min, y_max)] for j in range(x_min, x_max)] for i in range(n)])
#     # bs = np.array([[[pxArr[i][j, k][2] for k in range(y_min, y_max)] for j in range(x_min, x_max)] for i in range(n)])

#     # if median_bool:
#     #     func = np.frompyfunc(vec_median_clr, 3*n, 1)
#     #     pixels = func(*rs, *gs, *bs)
#     #     for i in range(x_min, x_max):
#     #         for j in range(y_min, y_max):
#     #             pxs[i, j] = pixels[i - x_min, j - y_min]

#     clrs = [(0,0,0) for i in range(n)]
#     for i in range(x_min, x_max):
#         for j in range(y_min, y_max):
#             for k in range(n):
#                 clrs[k] = imgs[k].getpixel((i, j))
#             pxs[i, j] = median_clr(clrs, delta)

#     return imgf

# returns boolean array of pixels that are part of the sky in img
# starts from px0 and adds all pixels that are within tau squared color 
# distance from adjacent pixels. If a group of pixels not identified
# as sky are completely surrounded by sky at the end, the group joins
# the sky. Starts search by assuming px0 is part of sky, where 
# px0 = (x0, y0)
def get_sky(img, tau, px0):
    w = img.width
    h = img.height
    pxs = img.load()
    isSky = np.zeros((h, w), dtype=np.int8) # 0 for not marked, -1 for not sky, 1 for sky
    isSky[px0[1], px0[0]] = 1
    sky_dfs(pxs, isSky, tau, px0, w, h, pxs)

    # change all unmarked to not sky
    isSky =- 1 # now, if 0, sky, if negative, not sky

    # find encircled regions and convert to sky


def sky_dfs(pxs, isSky, tau, px, w, h, pclr):
    if isSky[px[1], px[0]] != 0:
        return
    clr = pxs[px[1], px[0]]
    if sq_dist(pclr, clr) <= tau:
        isSky[px[1], px[0]] = 1
        if px[0] - 1 >= 0:
            sky_dfs(pxs, isSky, tau, (px[0]-1, px[1]), w, h, clr)
        if px[1] - 1 >= 0:
            sky_dfs(pxs, isSky, tau, (px[0], px[1]-1), w, h, clr)
        if px[0] + 1 <= w - 1:
            sky_dfs(pxs, isSky, tau, (px[0]+1, px[1]), w, h, clr)
        if px[1] + 1 <= h - 1:
            sky_dfs(pxs, isSky, tau, (px[0], px[1]+1), w, h, clr)
    else:
        isSky[px[1], px[0]] = -1


    
# main method, delegates tasks from arguments to other functions
def main(args):

    # read arguments and input
    out_format = args[1]
    if int(args[2]) == 0:
        write_file = False
    else: 
        write_file = True
    stack_method = int(args[3])
    # luminance case
    if stack_method == 0:
        if (int(args[4]) == 0):
            brightest = False
        else:
            brightest = True
        if (int(args[5]) == 0):
            parallel = False
        else:
            parallel = True
        n_args = 6
    # mean case
    elif stack_method == 1:
        n_args = 4
    # median case
    elif stack_method == 2:
        delta = int(args[4])
        n_args = 5
    else:
        print('Invalid stack_method argument')
        exit()

    if len(args) > n_args:
        x_range = (float(args[n_args]), float(args[n_args+1]))
        y_range = (float(args[n_args+2]), float(args[n_args+3]))
    else:
        x_range = (0, 1)
        y_range = (0, 1)

    if out_format != 'png' and out_format != 'jpg':
        print('Invalid output format argument')
        exit()

    out_name = 'img_stack.' + out_format
    if os.path.exists(out_name):
        print('Output file exists')
        exit()

    # read from standard input
    img_names = sys.stdin.readlines()

    start = time.time()
    if stack_method == 0:
        if parallel:
            imgf = lum_stack_par(img_names, x_range, y_range, brightest)
        else:
            imgf = lum_stack_bf(img_names, x_range, y_range, brightest)
    elif stack_method == 1:
        imgf = mean_stack(img_names, x_range, y_range)
    elif stack_method == 2:
        imgf = median_stack(img_names, x_range, y_range, delta)

    runtime = time.time() - start
    mins = runtime // 60
    secs = int(runtime % 60)

    print('Runtime: '+str(mins)+' minutes '+str(secs)+' seconds')

    if write_file:
        imgf.tofile('img_stack.bin', sep='')
    else:
        cv2.imwrite(out_name, imgf)

if __name__ == '__main__':  
    main(sys.argv)