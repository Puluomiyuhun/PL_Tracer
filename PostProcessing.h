#pragma once
#include <cuda_runtime.h>
struct BGR
{
    float b;
    float g;
    float r;
};

struct HSV
{
    int h;
    float s;
    float v;
};

extern "C" __device__ bool IsEquals(float val1, float val2)
{
    return val1 > val2 ? val1 - val2 < 0.001f : val2 - val1 < 0.001f;
}

extern "C" __device__ float MAX(float a, float b) {
    return a > b ? a : b;
}
extern "C" __device__ float MIN(float a, float b) {
    return a < b ? a : b;
}

// BGR(BGR: 0~255)תHSV(H: [0~360), S: [0~1], V: [0~1])
extern "C" __device__ void BGR2HSV(BGR& bgr, HSV& hsv)
{
    float b, g, r;
    float h, s, v;
    float min, max;
    float delta;
    b = bgr.b;
    g = bgr.g;
    r = bgr.r;
    if (r > g)
    {
        max = MAX(r, b);
        min = MIN(g, b);
    }
    else
    {
        max = MAX(g, b);
        min = MIN(r, b);
    }
    v = max;
    delta = max - min;

    if (IsEquals(max, 0))
        s = 0.0;
    else
        s = delta / max;

    if (max == min)
        h = 0.0;
    else{
        if (IsEquals(r, max) && g >= b)
            h = 60 * (g - b) / delta + 0;
        else if (IsEquals(r, max) && g < b)
            h = 60 * (g - b) / delta + 360;
        else if (IsEquals(g, max))
            h = 60 * (b - r) / delta + 120;
        else if (IsEquals(b, max))
            h = 60 * (r - g) / delta + 240;
    }
    hsv.h = (int)(h + 0.5);
    hsv.h = (hsv.h > 359) ? (hsv.h - 360) : hsv.h;
    hsv.h = (hsv.h < 0) ? (hsv.h + 360) : hsv.h;
    hsv.s = s;
    hsv.v = v;
}

// HSVתBGR
extern "C" __device__ void HSV2BGR(HSV& hsv, BGR& bgr)
{
    int h = hsv.h;
    float s = hsv.s;
    float v = hsv.v;
    if (s > 1.0f)s = 1.0f;
    if (v > 1.0f)v = 1.0f;
    float b = 0.0;
    float g = 0.0;
    float r = 0.0;

    int flag = (int)(h / 60.0);
    float f = h / 60.0 - flag;
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f) * s);
    switch (flag)
    {
    case 0:
        b = p;
        g = t;
        r = v;
        break;
    case 1:
        b = p;
        g = v;
        r = q;
        break;
    case 2:
        b = t;
        g = v;
        r = p;
        break;
    case 3:
        b = v;
        g = q;
        r = p;
        break;
    case 4:
        b = v;
        g = p;
        r = t;
        break;
    case 5:
        b = q;
        g = p;
        r = v;
        break;
    default:
        break;
    }

    bgr.b = b < 1.0f ? b : 1.0f;
    bgr.g = g < 1.0f ? g : 1.0f;
    bgr.r = r < 1.0f ? r : 1.0f;
}

extern "C" __device__ void Contrast(BGR & bgr, float con, float thre)
{
    bgr.r = bgr.r + (bgr.r - thre) * con;
    bgr.g = bgr.g + (bgr.g - thre) * con;
    bgr.b = bgr.b + (bgr.b - thre) * con;

    bgr.b = bgr.b < 1.0f ? bgr.b : 1.0f;
    bgr.g = bgr.g < 1.0f ? bgr.g : 1.0f;
    bgr.r = bgr.r < 1.0f ? bgr.r : 1.0f;

    bgr.b = bgr.b > 0.0f ? bgr.b : 0.0f;
    bgr.g = bgr.g > 0.0f ? bgr.g : 0.0f;
    bgr.r = bgr.r > 0.0f ? bgr.r : 0.0f;
}