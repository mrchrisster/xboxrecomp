/* xr_theater.c -- the game on a floating screen in a Quest headset; see
 * xr_theater.h. Ported from FuturamaRecomp's src/platform/xr/OpenXrTheater.cpp
 * (the placement, window mode, menu, Touch mapping and haptics), drawn here on
 * the GPU from the toolkit's own Vulkan device. */
#if defined(__ANDROID__) && defined(XBOXRECOMP_XR)

#define XR_USE_PLATFORM_ANDROID
#define XR_USE_GRAPHICS_API_VULKAN

#include <jni.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <vulkan/vulkan.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <SDL.h>
#include <SDL_system.h>

#include "d3d/nv2a_vk_core.h"
#include "input/xinput_xbox.h"
#include "input/touch_pad.h"
#include "xr_theater.h"

#define LOGX "[XR] "

/* ---- maths (local space, metres) ---- */
typedef XrVector3f V3;
static V3 v3(float x, float y, float z) { V3 r = {x, y, z}; return r; }
static V3 vadd(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static V3 vsub(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static V3 vmul(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static float vdot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 vcross(V3 a, V3 b) { return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static float vlen(V3 a) { return sqrtf(vdot(a, a)); }
static V3 vnorm(V3 a, V3 fb) { float l = vlen(a); return l > 1e-5f ? vmul(a, 1.0f / l) : fb; }
static float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }
static V3 qrot(XrQuaternionf q, V3 v)
{
    V3 u = v3(q.x, q.y, q.z), t = vmul(vcross(u, v), 2.0f);
    return vadd(vadd(v, vmul(t, q.w)), vcross(u, t));
}
static XrQuaternionf quat_from_to(V3 a, V3 b)
{
    XrQuaternionf r = {0, 0, 0, 1};
    float d = clampf(vdot(a, b), -1.0f, 1.0f), ang, s;
    V3 axis;
    if (d > 0.99999f)
        return r;
    axis = vcross(a, b);
    if (vlen(axis) < 1e-5f)
        axis = v3(0, 1, 0);
    axis = vnorm(axis, v3(0, 1, 0));
    ang = acosf(d);
    s = sinf(ang * 0.5f);
    r.x = axis.x * s; r.y = axis.y * s; r.z = axis.z * s; r.w = cosf(ang * 0.5f);
    return r;
}
/* A quad whose front (+Z) faces back along dir (head -> screen): yaw, pitch. */
static XrQuaternionf facing(V3 dir)
{
    float yaw = atan2f(-dir.x, -dir.z), pitch = asinf(clampf(dir.y, -1.0f, 1.0f));
    XrQuaternionf qy = {0, sinf(yaw * 0.5f), 0, cosf(yaw * 0.5f)};
    XrQuaternionf qp = {sinf(pitch * 0.5f), 0, 0, cosf(pitch * 0.5f)}, r;
    r.x = qy.w * qp.x + qy.x * qp.w + qy.y * qp.z - qy.z * qp.y;
    r.y = qy.w * qp.y - qy.x * qp.z + qy.y * qp.w + qy.z * qp.x;
    r.z = qy.w * qp.z + qy.x * qp.y - qy.y * qp.x + qy.z * qp.w;
    r.w = qy.w * qp.w - qy.x * qp.x - qy.y * qp.y - qy.z * qp.z;
    return r;
}

/* Defaults: Christoph's Futurama placement (2026-11-17) at 16:9. */
#define DEF_DIST   1.8f
#define DEF_ELEV   0.077f
#define DEF_HEIGHT 1.20f
#define DEF_DARK   1
#define MIN_DIST 0.6f
#define MAX_DIST 6.0f
#define MIN_H 0.4f
#define MAX_H 5.0f
#define STICK_DZ 0.3f

#define MENU_W 960
#define MENU_H 720
#define PROJ_DIM 64
#define MAXIMG 8

enum { M_RESUME, M_BACKGROUND, M_RECENTER, M_RESET, M_EXIT, M_COUNT };

static struct {
    int on;
    jobject activity;
    XrInstance inst;
    XrSystemId sys;
    XrSession session;
    XrSessionState state;
    int running;
    int ext_passthrough, ext_colorspace, ext_layer_settings, ext_perf, ext_refresh;
    int ext_touch_plus, ext_touch_pro;
    const struct nvk_core *C;
    VkCommandPool pool;
    VkCommandBuffer cb;
    VkFence fence;

    /* the screen's quad: written by the presenter, shown by the frame thread */
    pthread_mutex_t quad_mu;
    XrSwapchain quad;
    VkImage quad_img[MAXIMG];
    uint32_t quad_n, quad_w, quad_h, quad_idx;
    VkFormat quad_fmt;
    int quad_acquired, quad_has_frame;

    /* the eyes' projection layer: a 64x64 clear (black, or clear over passthrough) */
    XrSwapchain eye_sc[2];
    VkImage eye_img[2][MAXIMG];
    uint32_t eye_n[2];
    int64_t fmt_srgb;          /* the runtime's sRGB RGBA8 format */
    int64_t fmt_srgb_bgra;     /* and BGRA8, 0 if not offered */

    /* the VR menu panel, drawn on the CPU (rare), copied in from a buffer */
    XrSwapchain menu_sc;
    VkImage menu_img[MAXIMG];
    uint32_t menu_n;
    VkBuffer menu_buf;
    VkDeviceMemory menu_mem;
    uint8_t *menu_map;
    int menu_open, menu_sel, menu_dirty, exit_armed, menu_held, start_pulse;
    int prev_nav[6];

    XrSpace local, view, aim[2];
    XrPosef quad_pose;
    XrExtent2Df quad_size;
    float dist, height, elev;
    int dark;
    int grab, view_dirty;
    V3 grab_dir, grab_off;
    float grab_hand, grab_dist;
    int prev_bg, prev_reset, prev_menu_chord;
    int recenter;
    char prefs[512];

    /* passthrough (XR_FB_passthrough) */
    XrPassthroughFB pt;
    XrPassthroughLayerFB pt_layer;
    PFN_xrPassthroughLayerPauseFB pt_pause;
    PFN_xrPassthroughLayerResumeFB pt_resume;

    /* input */
    XrActionSet aset;
    XrPath hand[2];
    XrAction ls, rs, a, b, x, y, lt, rt, lg, rg, l3, r3, menu, haptic, aim_pose;
    int profile_logged;
    pthread_mutex_t pad_mu;
    XBOX_INPUT_STATE pad;
    int pad_active;
    unsigned motor[2], last_hap[2];
    uint64_t last_hap_ms[2];

    XrFrameState fs;
    pthread_t thread;
    int thread_run;
} X;

static uint64_t now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

static int has_ext(const XrExtensionProperties *e, uint32_t n, const char *name)
{
    uint32_t i;
    for (i = 0; i < n; i++)
        if (!strcmp(e[i].extensionName, name))
            return 1;
    return 0;
}

static PFN_xrVoidFunction xr_fn(const char *name)
{
    PFN_xrVoidFunction f = NULL;
    xrGetInstanceProcAddr(X.inst, name, &f);
    return f;
}

/* ---- the Vulkan instance and device, made by the runtime (enable2) ---- */
static VkResult hook_instance(const VkInstanceCreateInfo *ci, VkInstance *out)
{
    PFN_xrCreateVulkanInstanceKHR mk = (PFN_xrCreateVulkanInstanceKHR)xr_fn("xrCreateVulkanInstanceKHR");
    PFN_xrGetVulkanGraphicsRequirements2KHR req =
        (PFN_xrGetVulkanGraphicsRequirements2KHR)xr_fn("xrGetVulkanGraphicsRequirements2KHR");
    XrGraphicsRequirementsVulkan2KHR rq = {XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR};
    XrVulkanInstanceCreateInfoKHR xi = {XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR};
    VkResult vr = VK_SUCCESS;
    XrResult r;

    if (!mk || !req)
        return VK_ERROR_INITIALIZATION_FAILED;
    req(X.inst, X.sys, &rq);   /* must be asked before the instance is made */
    fprintf(stderr, LOGX "runtime Vulkan versions %u.%u .. %u.%u\n",
            (unsigned)XR_VERSION_MAJOR(rq.minApiVersionSupported), (unsigned)XR_VERSION_MINOR(rq.minApiVersionSupported),
            (unsigned)XR_VERSION_MAJOR(rq.maxApiVersionSupported), (unsigned)XR_VERSION_MINOR(rq.maxApiVersionSupported));
    xi.systemId = X.sys;
    xi.pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
    xi.vulkanCreateInfo = ci;
    r = mk(X.inst, &xi, out, &vr);
    if (XR_FAILED(r)) {
        fprintf(stderr, LOGX "xrCreateVulkanInstanceKHR: %d (vk %d)\n", (int)r, (int)vr);
        return vr != VK_SUCCESS ? vr : VK_ERROR_INITIALIZATION_FAILED;
    }
    return vr;
}

static VkPhysicalDevice hook_phys(VkInstance inst)
{
    PFN_xrGetVulkanGraphicsDevice2KHR get = (PFN_xrGetVulkanGraphicsDevice2KHR)xr_fn("xrGetVulkanGraphicsDevice2KHR");
    XrVulkanGraphicsDeviceGetInfoKHR gi = {XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR};
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    if (!get)
        return VK_NULL_HANDLE;
    gi.systemId = X.sys;
    gi.vulkanInstance = inst;
    if (XR_FAILED(get(X.inst, &gi, &pd)))
        return VK_NULL_HANDLE;
    return pd;
}

static VkResult hook_device(VkPhysicalDevice pd, const VkDeviceCreateInfo *ci, VkDevice *out)
{
    PFN_xrCreateVulkanDeviceKHR mk = (PFN_xrCreateVulkanDeviceKHR)xr_fn("xrCreateVulkanDeviceKHR");
    XrVulkanDeviceCreateInfoKHR xd = {XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR};
    VkResult vr = VK_SUCCESS;
    XrResult r;
    if (!mk)
        return VK_ERROR_INITIALIZATION_FAILED;
    xd.systemId = X.sys;
    xd.pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
    xd.vulkanPhysicalDevice = pd;
    xd.vulkanCreateInfo = ci;
    r = mk(X.inst, &xd, out, &vr);
    if (XR_FAILED(r)) {
        fprintf(stderr, LOGX "xrCreateVulkanDeviceKHR: %d (vk %d)\n", (int)r, (int)vr);
        return vr != VK_SUCCESS ? vr : VK_ERROR_INITIALIZATION_FAILED;
    }
    return vr;
}

int xrt_on(void) { return X.on; }

int xrt_preinit(void)
{
    static const struct nvk_xr_hooks hooks = {hook_instance, hook_phys, hook_device};
    const char *off = getenv("RECOMP_XR");
    JNIEnv *env;
    JavaVM *vm = NULL;
    jobject act;
    PFN_xrInitializeLoaderKHR init = NULL;
    XrLoaderInitInfoAndroidKHR li = {XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR};
    XrExtensionProperties *exts;
    uint32_t n = 0;
    const char *want[12];
    uint32_t nw = 0;
    XrInstanceCreateInfoAndroidKHR ia = {XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR};
    XrInstanceCreateInfo ici = {XR_TYPE_INSTANCE_CREATE_INFO};
    XrSystemGetInfo sgi = {XR_TYPE_SYSTEM_GET_INFO};
    XrResult r;

    if (X.on)
        return 1;
    if (off && off[0] == '0') {
        fprintf(stderr, LOGX "off (RECOMP_XR=0): the flat window\n");
        return 0;
    }
    env = (JNIEnv *)SDL_AndroidGetJNIEnv();
    if (!env || (*env)->GetJavaVM(env, &vm) != JNI_OK || !vm)
        return 0;
    act = (jobject)SDL_AndroidGetActivity();
    if (!act)
        return 0;
    X.activity = (*env)->NewGlobalRef(env, act);
    (*env)->DeleteLocalRef(env, act);

    xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR", (PFN_xrVoidFunction *)&init);
    if (!init)
        return 0;
    li.applicationVM = vm;
    li.applicationContext = X.activity;
    if (XR_FAILED(init((const XrLoaderInitInfoBaseHeaderKHR *)&li))) {
        fprintf(stderr, LOGX "no OpenXR loader here: the flat window\n");
        return 0;
    }
    xrEnumerateInstanceExtensionProperties(NULL, 0, &n, NULL);
    exts = (XrExtensionProperties *)calloc(n ? n : 1, sizeof *exts);
    if (!exts)
        return 0;
    for (uint32_t i = 0; i < n; i++)
        exts[i].type = XR_TYPE_EXTENSION_PROPERTIES;
    xrEnumerateInstanceExtensionProperties(NULL, n, &n, exts);
    if (!has_ext(exts, n, XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME)
            || !has_ext(exts, n, XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME)) {
        fprintf(stderr, LOGX "the runtime has no Vulkan enable2: the flat window\n");
        free(exts);
        return 0;
    }
    want[nw++] = XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME;
    want[nw++] = XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME;
#define OPT(flag, name) if (has_ext(exts, n, name)) { want[nw++] = name; X.flag = 1; }
    OPT(ext_passthrough, XR_FB_PASSTHROUGH_EXTENSION_NAME)
    OPT(ext_colorspace, XR_FB_COLOR_SPACE_EXTENSION_NAME)
    OPT(ext_layer_settings, XR_FB_COMPOSITION_LAYER_SETTINGS_EXTENSION_NAME)
    OPT(ext_perf, XR_EXT_PERFORMANCE_SETTINGS_EXTENSION_NAME)
    OPT(ext_refresh, XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME)
    OPT(ext_touch_plus, "XR_META_touch_controller_plus")
    OPT(ext_touch_pro, "XR_FB_touch_controller_pro")
#undef OPT
    free(exts);
    ia.applicationVM = vm;
    ia.applicationActivity = X.activity;
    ici.next = &ia;
    snprintf(ici.applicationInfo.applicationName, sizeof ici.applicationInfo.applicationName, "Takedawn");
    snprintf(ici.applicationInfo.engineName, sizeof ici.applicationInfo.engineName, "xboxrecomp");
    ici.applicationInfo.applicationVersion = 1;
    ici.applicationInfo.engineVersion = 1;
    ici.applicationInfo.apiVersion = XR_API_VERSION_1_0;
    ici.enabledExtensionCount = nw;
    ici.enabledExtensionNames = want;
    r = xrCreateInstance(&ici, &X.inst);
    if (XR_FAILED(r)) {
        fprintf(stderr, LOGX "xrCreateInstance: %d: the flat window\n", (int)r);
        return 0;
    }
    sgi.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    r = xrGetSystem(X.inst, &sgi, &X.sys);
    if (XR_FAILED(r)) {
        fprintf(stderr, LOGX "no headset (xrGetSystem %d): the flat window\n", (int)r);
        xrDestroyInstance(X.inst);
        X.inst = XR_NULL_HANDLE;
        return 0;
    }
    nvk_core_set_xr_hooks(&hooks);
    X.on = 1;
    fprintf(stderr, LOGX "OpenXR instance up (passthrough %d, refresh %d, touch plus %d): the Vulkan device"
            " comes from the runtime\n", X.ext_passthrough, X.ext_refresh, X.ext_touch_plus);
    return 1;
}

/* ---- the placement, kept in xr_view.txt: "t1 dist height elev dark" ---- */
static void update_size(void)
{
    float aspect = X.quad_h ? (float)X.quad_w / (float)X.quad_h : 16.0f / 9.0f;
    X.quad_size.height = X.height;
    X.quad_size.width = X.height * aspect;
}

static void load_view(void)
{
    const char *dir = SDL_AndroidGetExternalStoragePath();
    FILE *f;
    char tag[8] = {0};
    float d, h, e;
    int dk;
    X.dist = DEF_DIST;
    X.height = DEF_HEIGHT;
    X.elev = DEF_ELEV;
    X.dark = DEF_DARK;
    if (!dir)
        return;
    snprintf(X.prefs, sizeof X.prefs, "%s/xr_view.txt", dir);
    f = fopen(X.prefs, "r");
    if (!f)
        return;
    if (fscanf(f, "%7s %f %f %f %d", tag, &d, &h, &e, &dk) == 5 && !strcmp(tag, "t1")) {
        X.dist = clampf(d, MIN_DIST, MAX_DIST);
        X.height = clampf(h, MIN_H, MAX_H);
        X.elev = clampf(e, -1.2f, 1.2f);
        X.dark = dk != 0;
        fprintf(stderr, LOGX "view restored: %.2f m away, %.2f m tall, elevation %.2f, %s\n", X.dist,
                X.height, X.elev, X.dark ? "dark room" : "mixed reality");
    }
    fclose(f);
}

static void save_view(void)
{
    FILE *f;
    X.view_dirty = 0;
    if (!X.prefs[0] || !(f = fopen(X.prefs, "w")))
        return;
    fprintf(f, "t1 %.3f %.3f %.3f %d\n", X.dist, X.height, X.elev, X.dark);
    fclose(f);
}

static void apply_background(void)
{
    if (!X.pt_layer)
        return;
    if (X.dark) {
        if (X.pt_pause)
            X.pt_pause(X.pt_layer);
    } else if (X.pt_resume) {
        X.pt_resume(X.pt_layer);
    }
    fprintf(stderr, LOGX "background: %s\n", X.dark ? "dark room" : "mixed reality");
}

static void reset_view(void)
{
    X.dist = DEF_DIST;
    X.height = DEF_HEIGHT;
    X.elev = DEF_ELEV;
    update_size();
    X.grab = 0;
    X.recenter = 1;
    X.view_dirty = 1;
}

#define LOC_OK (XR_SPACE_LOCATION_ORIENTATION_VALID_BIT | XR_SPACE_LOCATION_POSITION_VALID_BIT)

static void do_recenter(XrTime t)
{
    XrSpaceLocation loc = {XR_TYPE_SPACE_LOCATION};
    const XrQuaternionf *q;
    float fx, fz, len, ce, se;
    V3 dir;
    if (t <= 0 || XR_FAILED(xrLocateSpace(X.view, X.local, t, &loc)) || (loc.locationFlags & LOC_OK) != LOC_OK)
        return;
    q = &loc.pose.orientation;
    fx = -2.0f * (q->x * q->z + q->w * q->y);
    fz = -(1.0f - 2.0f * (q->x * q->x + q->y * q->y));
    len = hypotf(fx, fz);
    if (len < 1e-4f) { fx = 0; fz = -1; } else { fx /= len; fz /= len; }
    ce = cosf(X.elev);
    se = sinf(X.elev);
    dir = v3(fx * ce, se, fz * ce);
    X.quad_pose.position = vadd(loc.pose.position, vmul(dir, X.dist));
    X.quad_pose.orientation = facing(dir);
    X.recenter = 0;
}

/* Both grips: the screen follows the hands (orbits the head), hands apart
 * push it away, L stick Y distance, R stick Y size. Saved on release. */
static void update_grab(int active, float dstick, float sstick, XrTime t)
{
    XrSpaceLocation head = {XR_TYPE_SPACE_LOCATION}, hl = {XR_TYPE_SPACE_LOCATION}, hr = {XR_TYPE_SPACE_LOCATION};
    V3 mid, aimv, pt, dir, offdir;
    float hd, ratio, nd;
    if (!active) {
        if (X.grab) {
            X.grab = 0;
            X.view_dirty = 1;
        }
        if (X.view_dirty)
            save_view();
        return;
    }
    if (t <= 0 || !X.aim[0] || !X.aim[1])
        return;
    if (XR_FAILED(xrLocateSpace(X.view, X.local, t, &head)) || (head.locationFlags & LOC_OK) != LOC_OK
            || XR_FAILED(xrLocateSpace(X.aim[0], X.local, t, &hl)) || (hl.locationFlags & LOC_OK) != LOC_OK
            || XR_FAILED(xrLocateSpace(X.aim[1], X.local, t, &hr)) || (hr.locationFlags & LOC_OK) != LOC_OK)
        return;
    mid = vmul(vadd(hl.pose.position, hr.pose.position), 0.5f);
    aimv = vnorm(vadd(qrot(hl.pose.orientation, v3(0, 0, -1)), qrot(hr.pose.orientation, v3(0, 0, -1))), v3(0, 0, -1));
    pt = vadd(mid, vmul(aimv, 0.38f));
    dir = vnorm(vsub(pt, head.pose.position), v3(0, 0, -1));
    hd = vlen(vsub(hl.pose.position, hr.pose.position));
    if (!X.grab) {
        X.grab = 1;
        X.grab_dir = dir;
        X.grab_hand = hd > 0.12f ? hd : 0.12f;
        X.grab_off = vsub(X.quad_pose.position, head.pose.position);
        if (vlen(X.grab_off) < 0.2f)
            X.grab_off = vmul(dir, X.dist);
        X.grab_dist = clampf(vlen(X.grab_off), MIN_DIST, MAX_DIST);
    }
    if (fabsf(dstick) > STICK_DZ)
        X.grab_dist = clampf(X.grab_dist + dstick * 0.025f, MIN_DIST, MAX_DIST);
    if (fabsf(sstick) > STICK_DZ) {
        X.height = clampf(X.height * (1.0f + sstick * 0.015f), MIN_H, MAX_H);
        update_size();
    }
    ratio = clampf(hd / X.grab_hand, 0.5f, 2.0f);
    nd = clampf(X.grab_dist * ratio, MIN_DIST, MAX_DIST);
    offdir = vnorm(qrot(quat_from_to(X.grab_dir, dir), vnorm(X.grab_off, dir)), dir);
    X.quad_pose.position = vadd(head.pose.position, vmul(offdir, nd));
    X.quad_pose.orientation = facing(offdir);
    X.dist = nd;
    X.elev = asinf(clampf(offdir.y, -1.0f, 1.0f));
}

/* ---- the VR menu (RGBA8 rows top-down, a 5x7 font) ---- */
static const struct { char ch; uint8_t col[5]; } k_font[] = {
    {' ', {0x00, 0x00, 0x00, 0x00, 0x00}}, {':', {0x00, 0x36, 0x36, 0x00, 0x00}},
    {'/', {0x20, 0x10, 0x08, 0x04, 0x02}}, {'-', {0x08, 0x08, 0x08, 0x08, 0x08}},
    {'+', {0x08, 0x08, 0x3E, 0x08, 0x08}}, {'.', {0x00, 0x60, 0x60, 0x00, 0x00}},
    {'!', {0x00, 0x00, 0x5F, 0x00, 0x00}}, {'=', {0x14, 0x14, 0x14, 0x14, 0x14}},
    {'0', {0x3E, 0x51, 0x49, 0x45, 0x3E}}, {'1', {0x00, 0x42, 0x7F, 0x40, 0x00}},
    {'2', {0x42, 0x61, 0x51, 0x49, 0x46}}, {'3', {0x21, 0x41, 0x45, 0x4B, 0x31}},
    {'4', {0x18, 0x14, 0x12, 0x7F, 0x10}}, {'5', {0x27, 0x45, 0x45, 0x45, 0x39}},
    {'6', {0x3C, 0x4A, 0x49, 0x49, 0x30}}, {'7', {0x01, 0x71, 0x09, 0x05, 0x03}},
    {'8', {0x36, 0x49, 0x49, 0x49, 0x36}}, {'9', {0x06, 0x49, 0x49, 0x29, 0x1E}},
    {'A', {0x7E, 0x11, 0x11, 0x11, 0x7E}}, {'B', {0x7F, 0x49, 0x49, 0x49, 0x36}},
    {'C', {0x3E, 0x41, 0x41, 0x41, 0x22}}, {'D', {0x7F, 0x41, 0x41, 0x22, 0x1C}},
    {'E', {0x7F, 0x49, 0x49, 0x49, 0x41}}, {'F', {0x7F, 0x09, 0x09, 0x09, 0x01}},
    {'G', {0x3E, 0x41, 0x49, 0x49, 0x7A}}, {'H', {0x7F, 0x08, 0x08, 0x08, 0x7F}},
    {'I', {0x00, 0x41, 0x7F, 0x41, 0x00}}, {'J', {0x20, 0x40, 0x41, 0x3F, 0x01}},
    {'K', {0x7F, 0x08, 0x14, 0x22, 0x41}}, {'L', {0x7F, 0x40, 0x40, 0x40, 0x40}},
    {'M', {0x7F, 0x02, 0x0C, 0x02, 0x7F}}, {'N', {0x7F, 0x04, 0x08, 0x10, 0x7F}},
    {'O', {0x3E, 0x41, 0x41, 0x41, 0x3E}}, {'P', {0x7F, 0x09, 0x09, 0x09, 0x06}},
    {'Q', {0x3E, 0x41, 0x51, 0x21, 0x5E}}, {'R', {0x7F, 0x09, 0x19, 0x29, 0x46}},
    {'S', {0x46, 0x49, 0x49, 0x49, 0x31}}, {'T', {0x01, 0x01, 0x7F, 0x01, 0x01}},
    {'U', {0x3F, 0x40, 0x40, 0x40, 0x3F}}, {'V', {0x1F, 0x20, 0x40, 0x20, 0x1F}},
    {'W', {0x3F, 0x40, 0x38, 0x40, 0x3F}}, {'X', {0x63, 0x14, 0x08, 0x14, 0x63}},
    {'Y', {0x07, 0x08, 0x70, 0x08, 0x07}}, {'Z', {0x61, 0x51, 0x49, 0x45, 0x43}},
};

static const uint8_t *glyph(char c)
{
    size_t i;
    if (c >= 'a' && c <= 'z')
        c = (char)(c - 'a' + 'A');
    for (i = 0; i < sizeof k_font / sizeof k_font[0]; i++)
        if (k_font[i].ch == c)
            return k_font[i].col;
    return k_font[0].col;
}

static void mfill(uint8_t *img, int x0, int y0, int w, int h, uint32_t rgba)
{
    int x, y;
    for (y = y0 < 0 ? 0 : y0; y < MENU_H && y < y0 + h; y++)
        for (x = x0 < 0 ? 0 : x0; x < MENU_W && x < x0 + w; x++) {
            uint8_t *d = img + ((size_t)y * MENU_W + x) * 4;
            d[0] = (uint8_t)(rgba >> 24); d[1] = (uint8_t)(rgba >> 16);
            d[2] = (uint8_t)(rgba >> 8); d[3] = (uint8_t)rgba;
        }
}

static void mtext(uint8_t *img, int x0, int y0, int s, uint32_t rgba, const char *t)
{
    int cx = x0, gx, gy;
    for (; *t; t++, cx += 6 * s) {
        const uint8_t *c = glyph(*t);
        for (gx = 0; gx < 5; gx++)
            for (gy = 0; gy < 7; gy++)
                if (c[gx] & (1u << gy))
                    mfill(img, cx + gx * s, y0 + gy * s, s, s, rgba);
    }
}

static int mwidth(const char *t, int s) { return (int)strlen(t) * 6 * s - s; }

static void draw_menu(uint8_t *img)
{
    static const char *labels[M_COUNT] = {"RESUME", "BACKGROUND", "RECENTER SCREEN", "RESET SCREEN", "EXIT GAME"};
    const char *values[M_COUNT] = {"", X.dark || !X.pt_layer ? "DARK ROOM" : "MIXED REALITY", "", "",
                                   X.exit_armed ? "PRESS A AGAIN" : ""};
    const int band = 170, top = band + 6 + 40;
    int i;
    mfill(img, 0, 0, MENU_W, MENU_H, 0xF26B1DFFu);              /* orange border */
    mfill(img, 6, 6, MENU_W - 12, MENU_H - 12, 0x101014FFu);     /* dark panel */
    mfill(img, 6, 6, MENU_W - 12, band, 0x2A0F08FFu);            /* header band */
    mtext(img, (MENU_W - mwidth("BURNOUT 3", 12)) / 2, 6 + 22, 12, 0xF26B1DFFu, "BURNOUT 3");
    mtext(img, (MENU_W - mwidth("TAKEDAWN VR", 4)) / 2, 6 + 22 + 96, 4, 0xFFD24AFFu, "TAKEDAWN VR");
    for (i = 0; i < M_COUNT; i++) {
        int y = top + i * 70, sel = i == X.menu_sel;
        uint32_t fg = sel ? 0x101014FFu : 0xFFFFFFFFu;
        if (sel)
            mfill(img, 20, y - 10, MENU_W - 40, 58, 0xF26B1DFFu);
        mtext(img, 44, y, 5, fg, labels[i]);
        if (values[i][0]) {
            char v[64];
            snprintf(v, sizeof v, i == M_EXIT ? "%s" : "- %s +", values[i]);
            mtext(img, MENU_W - 44 - mwidth(v, 4), y + 4, 4, sel ? 0x101014FFu : 0xFFD24AFFu, v);
        }
    }
    mtext(img, 44, MENU_H - 50, 3, 0xB8C4E0FFu, "STICK: CHANGE  A: OK  B: CLOSE  L3+R3: CLOSE");
}

static void set_menu(int open)
{
    int i;
    X.menu_open = open;
    X.menu_dirty = 1;
    X.exit_armed = 0;
    if (open)
        X.menu_sel = 0;
    for (i = 0; i < 6; i++)
        X.prev_nav[i] = 1;     /* buttons already held when it opens are not presses */
}

static void menu_activate(int dir)
{
    X.menu_dirty = 1;
    switch (X.menu_sel) {
    case M_RESUME:
        if (dir == 0)
            set_menu(0);
        break;
    case M_BACKGROUND:
        if (!X.pt_layer)
            break;
        X.dark = !X.dark;
        apply_background();
        save_view();
        break;
    case M_RECENTER:
        if (dir == 0)
            X.recenter = 1;
        break;
    case M_RESET:
        if (dir == 0) {
            reset_view();
            save_view();
        }
        break;
    case M_EXIT:
        if (dir != 0)
            break;
        if (X.exit_armed) {
            SDL_Event ev;
            memset(&ev, 0, sizeof ev);
            ev.type = SDL_QUIT;
            SDL_PushEvent(&ev);
            fprintf(stderr, LOGX "exit from the VR menu\n");
            set_menu(0);
        } else {
            X.exit_armed = 1;
        }
        break;
    }
}

static void menu_input(int up, int down, int left, int right, int a, int b)
{
    int now[6] = {up, down, left, right, a, b}, e[6], i;
    for (i = 0; i < 6; i++) {
        e[i] = now[i] && !X.prev_nav[i];
        X.prev_nav[i] = now[i];
    }
    if (e[0]) { X.menu_sel = (X.menu_sel + M_COUNT - 1) % M_COUNT; X.exit_armed = 0; X.menu_dirty = 1; }
    if (e[1]) { X.menu_sel = (X.menu_sel + 1) % M_COUNT; X.exit_armed = 0; X.menu_dirty = 1; }
    if (e[2]) menu_activate(-1);
    if (e[3]) menu_activate(+1);
    if (e[4]) menu_activate(0);
    if (e[5]) set_menu(0);
}

/* ---- input: Touch -> the Xbox pad on port 0 ---- */
static int rd_bool(XrAction a)
{
    XrActionStateBoolean s = {XR_TYPE_ACTION_STATE_BOOLEAN};
    XrActionStateGetInfo gi = {XR_TYPE_ACTION_STATE_GET_INFO};
    gi.action = a;
    xrGetActionStateBoolean(X.session, &gi, &s);
    return s.isActive && s.currentState == XR_TRUE;
}
static float rd_float(XrAction a)
{
    XrActionStateFloat s = {XR_TYPE_ACTION_STATE_FLOAT};
    XrActionStateGetInfo gi = {XR_TYPE_ACTION_STATE_GET_INFO};
    gi.action = a;
    xrGetActionStateFloat(X.session, &gi, &s);
    return s.isActive ? clampf(s.currentState, 0.0f, 1.0f) : 0.0f;
}
static XrVector2f rd_vec2(XrAction a)
{
    XrActionStateVector2f s = {XR_TYPE_ACTION_STATE_VECTOR2F};
    XrActionStateGetInfo gi = {XR_TYPE_ACTION_STATE_GET_INFO};
    XrVector2f z = {0, 0};
    gi.action = a;
    xrGetActionStateVector2f(X.session, &gi, &s);
    return s.isActive ? s.currentState : z;
}
static SHORT axis16(float v)
{
    long l = lroundf(v * 32767.0f);
    return (SHORT)(l < -32768 ? -32768 : l > 32767 ? 32767 : l);
}

static void log_profiles(void)
{
    int i;
    X.profile_logged = 1;
    for (i = 0; i < 2; i++) {
        XrInteractionProfileState ips = {XR_TYPE_INTERACTION_PROFILE_STATE};
        char buf[XR_MAX_PATH_LENGTH] = "<none>";
        uint32_t len = 0;
        if (XR_SUCCEEDED(xrGetCurrentInteractionProfile(X.session, X.hand[i], &ips))
                && ips.interactionProfile != XR_NULL_PATH)
            xrPathToString(X.inst, ips.interactionProfile, sizeof buf, &len, buf);
        fprintf(stderr, LOGX "%s hand: %s\n", i ? "right" : "left", buf);
    }
}

static void sync_actions(void)
{
    XBOX_INPUT_STATE st;
    XrActiveActionSet aas;
    XrActionsSyncInfo asi = {XR_TYPE_ACTIONS_SYNC_INFO};
    XrVector2f ls, rs;
    float lt, rt, lg, rg;
    int lb, rb, l3, r3, menu, chord, window, bg, reset, tap, any;
    static DWORD packet;

    memset(&st, 0, sizeof st);
    if (X.state != XR_SESSION_STATE_FOCUSED)
        goto publish;
    if (!X.profile_logged)
        log_profiles();
    aas.actionSet = X.aset;
    aas.subactionPath = XR_NULL_PATH;
    asi.countActiveActionSets = 1;
    asi.activeActionSets = &aas;
    if (XR_FAILED(xrSyncActions(X.session, &asi)))
        goto publish;

    ls = rd_vec2(X.ls);
    rs = rd_vec2(X.rs);
    lt = rd_float(X.lt);
    rt = rd_float(X.rt);
    lg = rd_float(X.lg);
    rg = rd_float(X.rg);
    lb = lg > 0.5f;
    rb = rg > 0.5f;
    l3 = rd_bool(X.l3);
    r3 = rd_bool(X.r3);
    menu = rd_bool(X.menu);

    /* L3 + R3: the VR menu. (The runtime gives the Menu button only as a
     * short pulse on release, so no long press.) */
    chord = l3 && r3;
    if (chord && !X.prev_menu_chord)
        set_menu(!X.menu_open);
    X.prev_menu_chord = chord;

    window = lb && rb;
    bg = window && r3 && !l3;
    if (bg && !X.prev_bg) {
        X.dark = !X.dark;
        apply_background();
        X.view_dirty = 1;
        X.menu_dirty = 1;
    }
    X.prev_bg = bg;
    reset = window && rd_bool(X.y);
    if (reset && !X.prev_reset)
        reset_view();
    X.prev_reset = reset;
    if (!reset)
        update_grab(window, ls.y, rs.y, X.fs.predictedDisplayTime);
    tap = menu && !X.menu_held;
    X.menu_held = menu;
    if (window) {
        if (menu)
            st.Gamepad.wButtons |= 0x0020;           /* Back */
        goto publish;
    }
    if (X.menu_open) {                               /* the game gets nothing meanwhile */
        if (tap) {
            set_menu(0);
            goto publish;
        }
        menu_input(ls.y > 0.6f || rs.y > 0.6f, ls.y < -0.6f || rs.y < -0.6f,
                   ls.x < -0.6f || rs.x < -0.6f, ls.x > 0.6f || rs.x > 0.6f, rd_bool(X.a), rd_bool(X.b));
        goto publish;
    }
    if (tap)
        X.start_pulse = 6;                           /* the pulse held a few frames */
    if (X.start_pulse > 0) {
        st.Gamepad.wButtons |= 0x0010;               /* Start */
        X.start_pulse--;
    }
    if (l3) st.Gamepad.wButtons |= 0x0040;
    if (r3) st.Gamepad.wButtons |= 0x0080;
    st.Gamepad.bAnalogButtons[0] = rd_bool(X.a) ? 255 : 0;
    st.Gamepad.bAnalogButtons[1] = rd_bool(X.b) ? 255 : 0;
    st.Gamepad.bAnalogButtons[2] = rd_bool(X.x) ? 255 : 0;
    st.Gamepad.bAnalogButtons[3] = rd_bool(X.y) ? 255 : 0;
    st.Gamepad.bAnalogButtons[4] = rb ? 255 : 0;     /* Black */
    st.Gamepad.bAnalogButtons[5] = lb ? 255 : 0;     /* White */
    st.Gamepad.bAnalogButtons[6] = (BYTE)lroundf(lt * 255.0f);
    st.Gamepad.bAnalogButtons[7] = (BYTE)lroundf(rt * 255.0f);
    st.Gamepad.sThumbLX = axis16(ls.x);
    st.Gamepad.sThumbLY = axis16(ls.y);
    st.Gamepad.sThumbRX = axis16(rs.x);
    st.Gamepad.sThumbRY = axis16(rs.y);

publish:
    any = st.Gamepad.wButtons || st.Gamepad.sThumbLX || st.Gamepad.sThumbLY || st.Gamepad.sThumbRX
          || st.Gamepad.sThumbRY;
    for (int i = 0; i < 8; i++)
        any |= st.Gamepad.bAnalogButtons[i] > 32;
    pthread_mutex_lock(&X.pad_mu);
    if (memcmp(&st.Gamepad, &X.pad.Gamepad, sizeof st.Gamepad))
        packet++;
    st.dwPacketNumber = packet;
    X.pad = st;
    X.pad_active = X.state == XR_SESSION_STATE_FOCUSED;
    pthread_mutex_unlock(&X.pad_mu);
    if (any)
        xbox_InputTouchControllerUsed();   /* the on-screen controls hide */
}

static int xr_pad_state(XBOX_INPUT_STATE *out)
{
    int on;
    pthread_mutex_lock(&X.pad_mu);
    on = X.pad_active;
    *out = X.pad;
    pthread_mutex_unlock(&X.pad_mu);
    return on;
}

static void refresh_haptics(int force)
{
    uint64_t t = now_ms();
    int i;
    if (!X.running || X.state != XR_SESSION_STATE_FOCUSED || !X.haptic)
        return;
    for (i = 0; i < 2; i++) {
        XrHapticActionInfo hai = {XR_TYPE_HAPTIC_ACTION_INFO};
        unsigned v = X.motor[i];
        hai.action = X.haptic;
        hai.subactionPath = X.hand[i];
        if (!v) {
            if (X.last_hap[i] || force)
                xrStopHapticFeedback(X.session, &hai);
            X.last_hap[i] = 0;
            X.last_hap_ms[i] = 0;
            continue;
        }
        if (X.last_hap[i] && t - X.last_hap_ms[i] < 65
                && (!force || abs((int)v - (int)X.last_hap[i]) < 6554))
            continue;
        {
            XrHapticVibration vib = {XR_TYPE_HAPTIC_VIBRATION};
            vib.duration = 100000000LL;      /* 100 ms, renewed every ~65 ms */
            vib.frequency = XR_FREQUENCY_UNSPECIFIED;
            vib.amplitude = clampf((float)v / 65535.0f, 0.0f, 1.0f);
            xrApplyHapticFeedback(X.session, &hai, (const XrHapticBaseHeader *)&vib);
        }
        X.last_hap[i] = v;
        X.last_hap_ms[i] = t;
    }
}

/* The game's motors (any thread): picked up by the frame thread. Left (low
 * frequency) to the left hand, right to the right. */
static void xr_rumble(unsigned left, unsigned right)
{
    X.motor[0] = left;
    X.motor[1] = right;
}

static int make_actions(void)
{
    XrActionSetCreateInfo asci = {XR_TYPE_ACTION_SET_CREATE_INFO};
    XrSessionActionSetsAttachInfo ai = {XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
    XrPath p;
    struct { XrAction *a; const char *name; XrActionType type; int subs; } acts[] = {
        {&X.ls, "left_stick", XR_ACTION_TYPE_VECTOR2F_INPUT, 0},
        {&X.rs, "right_stick", XR_ACTION_TYPE_VECTOR2F_INPUT, 0},
        {&X.a, "btn_a", XR_ACTION_TYPE_BOOLEAN_INPUT, 0},
        {&X.b, "btn_b", XR_ACTION_TYPE_BOOLEAN_INPUT, 0},
        {&X.x, "btn_x", XR_ACTION_TYPE_BOOLEAN_INPUT, 0},
        {&X.y, "btn_y", XR_ACTION_TYPE_BOOLEAN_INPUT, 0},
        {&X.lt, "left_trigger", XR_ACTION_TYPE_FLOAT_INPUT, 0},
        {&X.rt, "right_trigger", XR_ACTION_TYPE_FLOAT_INPUT, 0},
        {&X.lg, "left_grip", XR_ACTION_TYPE_FLOAT_INPUT, 0},
        {&X.rg, "right_grip", XR_ACTION_TYPE_FLOAT_INPUT, 0},
        {&X.l3, "left_stick_click", XR_ACTION_TYPE_BOOLEAN_INPUT, 0},
        {&X.r3, "right_stick_click", XR_ACTION_TYPE_BOOLEAN_INPUT, 0},
        {&X.menu, "menu_click", XR_ACTION_TYPE_BOOLEAN_INPUT, 0},
        {&X.haptic, "haptic", XR_ACTION_TYPE_VIBRATION_OUTPUT, 1},
        {&X.aim_pose, "aim_pose", XR_ACTION_TYPE_POSE_INPUT, 1},
    };
    struct { XrAction *a; const char *path; } binds[] = {
        {&X.ls, "/user/hand/left/input/thumbstick"}, {&X.rs, "/user/hand/right/input/thumbstick"},
        {&X.a, "/user/hand/right/input/a/click"}, {&X.b, "/user/hand/right/input/b/click"},
        {&X.x, "/user/hand/left/input/x/click"}, {&X.y, "/user/hand/left/input/y/click"},
        {&X.lt, "/user/hand/left/input/trigger/value"}, {&X.rt, "/user/hand/right/input/trigger/value"},
        {&X.lg, "/user/hand/left/input/squeeze/value"}, {&X.rg, "/user/hand/right/input/squeeze/value"},
        {&X.l3, "/user/hand/left/input/thumbstick/click"}, {&X.r3, "/user/hand/right/input/thumbstick/click"},
        {&X.menu, "/user/hand/left/input/menu/click"},
        {&X.haptic, "/user/hand/left/output/haptic"}, {&X.haptic, "/user/hand/right/output/haptic"},
        {&X.aim_pose, "/user/hand/left/input/aim/pose"}, {&X.aim_pose, "/user/hand/right/input/aim/pose"},
    };
    XrActionSuggestedBinding sb[sizeof binds / sizeof binds[0]];
    const char *profiles[3] = {"/interaction_profiles/oculus/touch_controller",
                               X.ext_touch_plus ? "/interaction_profiles/meta/touch_controller_plus" : NULL,
                               X.ext_touch_pro ? "/interaction_profiles/facebook/touch_controller_pro" : NULL};
    size_t i;

    xrStringToPath(X.inst, "/user/hand/left", &X.hand[0]);
    xrStringToPath(X.inst, "/user/hand/right", &X.hand[1]);
    snprintf(asci.actionSetName, sizeof asci.actionSetName, "takedawn");
    snprintf(asci.localizedActionSetName, sizeof asci.localizedActionSetName, "Takedawn");
    if (XR_FAILED(xrCreateActionSet(X.inst, &asci, &X.aset)))
        return 0;
    for (i = 0; i < sizeof acts / sizeof acts[0]; i++) {
        XrActionCreateInfo aci = {XR_TYPE_ACTION_CREATE_INFO};
        snprintf(aci.actionName, sizeof aci.actionName, "%s", acts[i].name);
        snprintf(aci.localizedActionName, sizeof aci.localizedActionName, "%s", acts[i].name);
        aci.actionType = acts[i].type;
        if (acts[i].subs) {
            aci.countSubactionPaths = 2;
            aci.subactionPaths = X.hand;
        }
        if (XR_FAILED(xrCreateAction(X.aset, &aci, acts[i].a)))
            return 0;
    }
    for (i = 0; i < sizeof binds / sizeof binds[0]; i++) {
        xrStringToPath(X.inst, binds[i].path, &p);
        sb[i].action = *binds[i].a;
        sb[i].binding = p;
    }
    for (i = 0; i < 3; i++) {
        XrInteractionProfileSuggestedBinding s = {XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
        XrResult r;
        if (!profiles[i])
            continue;
        xrStringToPath(X.inst, profiles[i], &p);
        s.interactionProfile = p;
        s.suggestedBindings = sb;
        s.countSuggestedBindings = (uint32_t)(sizeof binds / sizeof binds[0]);
        r = xrSuggestInteractionProfileBindings(X.inst, &s);
        fprintf(stderr, LOGX "profile %s: %s\n", profiles[i], XR_SUCCEEDED(r) ? "bound" : "rejected");
        if (XR_FAILED(r) && i == 0)
            return 0;
    }
    ai.countActionSets = 1;
    ai.actionSets = &X.aset;
    if (XR_FAILED(xrAttachSessionActionSets(X.session, &ai)))
        return 0;
    for (i = 0; i < 2; i++) {
        XrActionSpaceCreateInfo s = {XR_TYPE_ACTION_SPACE_CREATE_INFO};
        s.action = X.aim_pose;
        s.subactionPath = X.hand[i];
        s.poseInActionSpace.orientation.w = 1.0f;
        if (XR_FAILED(xrCreateActionSpace(X.session, &s, &X.aim[i])))
            X.aim[i] = XR_NULL_HANDLE;
    }
    return 1;
}

static void init_passthrough(void)
{
    PFN_xrCreatePassthroughFB mk;
    PFN_xrPassthroughStartFB start;
    PFN_xrCreatePassthroughLayerFB mkl;
    XrPassthroughCreateInfoFB pci = {XR_TYPE_PASSTHROUGH_CREATE_INFO_FB};
    XrPassthroughLayerCreateInfoFB plci = {XR_TYPE_PASSTHROUGH_LAYER_CREATE_INFO_FB};
    if (!X.ext_passthrough)
        return;
    mk = (PFN_xrCreatePassthroughFB)xr_fn("xrCreatePassthroughFB");
    start = (PFN_xrPassthroughStartFB)xr_fn("xrPassthroughStartFB");
    mkl = (PFN_xrCreatePassthroughLayerFB)xr_fn("xrCreatePassthroughLayerFB");
    X.pt_resume = (PFN_xrPassthroughLayerResumeFB)xr_fn("xrPassthroughLayerResumeFB");
    X.pt_pause = (PFN_xrPassthroughLayerPauseFB)xr_fn("xrPassthroughLayerPauseFB");
    if (!mk || !start || !mkl || !X.pt_resume)
        return;
    if (XR_FAILED(mk(X.session, &pci, &X.pt)) || XR_FAILED(start(X.pt)))
        return;
    plci.passthrough = X.pt;
    plci.purpose = XR_PASSTHROUGH_LAYER_PURPOSE_RECONSTRUCTION_FB;
    plci.flags = XR_PASSTHROUGH_IS_RUNNING_AT_CREATION_BIT_FB;
    if (XR_FAILED(mkl(X.session, &plci, &X.pt_layer)))
        X.pt_layer = XR_NULL_HANDLE;
    apply_background();
}

/* 120 Hz shows the game's 60 twice each, evenly; else 90, else the default. */
static void perf_and_refresh(void)
{
    if (X.ext_perf) {
        PFN_xrPerfSettingsSetPerformanceLevelEXT perf =
            (PFN_xrPerfSettingsSetPerformanceLevelEXT)xr_fn("xrPerfSettingsSetPerformanceLevelEXT");
        if (perf) {
            perf(X.session, XR_PERF_SETTINGS_DOMAIN_CPU_EXT, XR_PERF_SETTINGS_LEVEL_SUSTAINED_HIGH_EXT);
            perf(X.session, XR_PERF_SETTINGS_DOMAIN_GPU_EXT, XR_PERF_SETTINGS_LEVEL_SUSTAINED_HIGH_EXT);
        }
    }
    if (X.ext_refresh) {
        PFN_xrEnumerateDisplayRefreshRatesFB en =
            (PFN_xrEnumerateDisplayRefreshRatesFB)xr_fn("xrEnumerateDisplayRefreshRatesFB");
        PFN_xrRequestDisplayRefreshRateFB rq =
            (PFN_xrRequestDisplayRefreshRateFB)xr_fn("xrRequestDisplayRefreshRateFB");
        float rates[16], pick = 0.0f;
        uint32_t n = 0, i;
        const char *want = getenv("RECOMP_XR_HZ");
        float w = want && *want ? (float)atof(want) : 120.0f;
        if (!en || !rq || XR_FAILED(en(X.session, 16, &n, rates)))
            return;
        for (i = 0; i < n; i++)
            if (fabsf(rates[i] - w) < 0.6f)
                pick = rates[i];
        if (!pick)
            for (i = 0; i < n; i++)
                if (fabsf(rates[i] - 60.0f) < 0.6f || fabsf(rates[i] - 90.0f) < 0.6f)
                    pick = rates[i];
        if (pick)
            fprintf(stderr, LOGX "display %.0f Hz (%s)\n", pick, XR_SUCCEEDED(rq(X.session, pick)) ? "set" : "refused");
    }
}

/* ---- swapchains ---- */
static uint32_t enum_images(XrSwapchain sc, VkImage *out)
{
    XrSwapchainImageVulkan2KHR im[MAXIMG];
    uint32_t n = 0, i;
    for (i = 0; i < MAXIMG; i++) {
        im[i].type = XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR;
        im[i].next = NULL;
    }
    xrEnumerateSwapchainImages(sc, 0, &n, NULL);
    if (n > MAXIMG)
        n = MAXIMG;
    xrEnumerateSwapchainImages(sc, n, &n, (XrSwapchainImageBaseHeader *)im);
    for (i = 0; i < n; i++)
        out[i] = im[i].image;
    return n;
}

static XrSwapchain make_sc(int64_t fmt, uint32_t w, uint32_t h, XrSwapchainUsageFlags usage)
{
    XrSwapchainCreateInfo ci = {XR_TYPE_SWAPCHAIN_CREATE_INFO};
    XrSwapchain sc = XR_NULL_HANDLE;
    ci.usageFlags = usage;
    ci.format = fmt;
    ci.sampleCount = 1;
    ci.width = w;
    ci.height = h;
    ci.faceCount = 1;
    ci.arraySize = 1;
    ci.mipCount = 1;
    if (XR_FAILED(xrCreateSwapchain(X.session, &ci, &sc)))
        return XR_NULL_HANDLE;
    return sc;
}

static void pick_formats(void)
{
    int64_t f[64];
    uint32_t n = 0, i;
    xrEnumerateSwapchainFormats(X.session, 64, &n, f);
    for (i = 0; i < n; i++) {
        if (f[i] == VK_FORMAT_R8G8B8A8_SRGB)
            X.fmt_srgb = f[i];
        if (f[i] == VK_FORMAT_B8G8R8A8_SRGB)
            X.fmt_srgb_bgra = f[i];
    }
    if (!X.fmt_srgb && n)
        X.fmt_srgb = f[0];
}

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want)
{
    uint32_t i;
    for (i = 0; i < X.C->mem.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (X.C->mem.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

static int make_menu(void)
{
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    VkMemoryRequirements mr;
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    VkDeviceSize sz = (VkDeviceSize)MENU_W * MENU_H * 4;
    bci.size = sz;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (vkCreateBuffer(X.C->dev, &bci, NULL, &X.menu_buf) != VK_SUCCESS)
        return 0;
    vkGetBufferMemoryRequirements(X.C->dev, X.menu_buf, &mr);
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                    | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(X.C->dev, &mai, NULL, &X.menu_mem) != VK_SUCCESS
            || vkBindBufferMemory(X.C->dev, X.menu_buf, X.menu_mem, 0) != VK_SUCCESS
            || vkMapMemory(X.C->dev, X.menu_mem, 0, sz, 0, (void **)&X.menu_map) != VK_SUCCESS)
        return 0;
    X.menu_sc = make_sc(VK_FORMAT_R8G8B8A8_SRGB, MENU_W, MENU_H,
                        XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT
                        | XR_SWAPCHAIN_USAGE_SAMPLED_BIT);
    if (!X.menu_sc)
        return 0;
    X.menu_n = enum_images(X.menu_sc, X.menu_img);
    return 1;
}

static void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to,
                    VkAccessFlags sa, VkAccessFlags da, VkPipelineStageFlags ss, VkPipelineStageFlags ds)
{
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = sa;
    b.dstAccessMask = da;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cb, ss, ds, 0, 0, NULL, 0, NULL, 1, &b);
}

/* The frame thread's own GPU work (the eyes' clears, the menu's upload), on
 * the shared queue: one command buffer, waited before reuse. */
static VkCommandBuffer own_begin(void)
{
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkWaitForFences(X.C->dev, 1, &X.fence, VK_TRUE, 1000000000ull);
    vkResetFences(X.C->dev, 1, &X.fence);
    vkResetCommandBuffer(X.cb, 0);
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(X.cb, &bi);
    return X.cb;
}

static void own_submit(void)
{
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    vkEndCommandBuffer(X.cb);
    si.commandBufferCount = 1;
    si.pCommandBuffers = &X.cb;
    nvk_queue_lock();
    if (vkQueueSubmit(X.C->queue, 1, &si, X.fence) == VK_SUCCESS)
        nvk_submit_note("xr frame thread");
    nvk_queue_unlock();
}

static int acquire(XrSwapchain sc, uint32_t *idx)
{
    XrSwapchainImageAcquireInfo ai = {XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    XrSwapchainImageWaitInfo wi = {XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    XrResult r;
    nvk_queue_lock();
    r = xrAcquireSwapchainImage(sc, &ai, idx);
    nvk_queue_unlock();
    if (XR_FAILED(r))
        return 0;
    wi.timeout = 100000000LL;
    if (XR_FAILED(xrWaitSwapchainImage(sc, &wi))) {
        XrSwapchainImageReleaseInfo ri = {XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        nvk_queue_lock();
        xrReleaseSwapchainImage(sc, &ri);
        nvk_queue_unlock();
        return 0;
    }
    return 1;
}

static void release_locked(XrSwapchain sc)
{
    XrSwapchainImageReleaseInfo ri = {XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    xrReleaseSwapchainImage(sc, &ri);
}

/* ---- the presenter's side ---- */
void xrt_quad_record(VkCommandBuffer cb, VkImage src, VkFormat fmt, int x, int y, int w, int h)
{
    VkImageCopy c;
    int64_t want;
    if (!X.on || !X.session || w <= 0 || h <= 0 || X.quad_acquired)
        return;
    want = (fmt == VK_FORMAT_B8G8R8A8_UNORM || fmt == VK_FORMAT_B8G8R8A8_SRGB) ? X.fmt_srgb_bgra : X.fmt_srgb;
    if (!want)
        return;      /* no matching format: a raw copy would swap the colours */
    pthread_mutex_lock(&X.quad_mu);
    if (!X.quad || X.quad_w != (uint32_t)w || X.quad_h != (uint32_t)h || X.quad_fmt != (VkFormat)want) {
        if (X.quad)
            xrDestroySwapchain(X.quad);
        X.quad_has_frame = 0;
        X.quad = make_sc(want, (uint32_t)w, (uint32_t)h,
                         XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT
                         | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT);
        X.quad_n = X.quad ? enum_images(X.quad, X.quad_img) : 0;
        X.quad_w = (uint32_t)w;
        X.quad_h = (uint32_t)h;
        X.quad_fmt = (VkFormat)want;
        update_size();
        fprintf(stderr, LOGX "screen swapchain %dx%d (%s)\n", w, h, X.quad ? "made" : "FAILED");
    }
    pthread_mutex_unlock(&X.quad_mu);
    if (!X.quad || !acquire(X.quad, &X.quad_idx) || X.quad_idx >= X.quad_n)
        return;
    X.quad_acquired = 1;
    barrier(cb, src, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
    barrier(cb, X.quad_img[X.quad_idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    memset(&c, 0, sizeof c);
    c.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    c.srcSubresource.layerCount = 1;
    c.dstSubresource = c.srcSubresource;
    c.srcOffset.x = x;
    c.srcOffset.y = y;
    c.extent.width = (uint32_t)w;
    c.extent.height = (uint32_t)h;
    c.extent.depth = 1;
    /* A copy, not a blit: the bytes are already sRGB-encoded, and the
     * quad's sRGB format shows them as they are. */
    vkCmdCopyImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, X.quad_img[X.quad_idx],
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    barrier(cb, X.quad_img[X.quad_idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    barrier(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
}

void xrt_quad_submitted(int ok)
{
    if (!X.quad_acquired)
        return;
    X.quad_acquired = 0;
    release_locked(X.quad);   /* the caller holds nvk_queue_lock */
    if (ok)
        X.quad_has_frame = 1;
}

/* ---- the frame thread ---- */
static void poll_events(void)
{
    XrEventDataBuffer ev = {XR_TYPE_EVENT_DATA_BUFFER};
    while (xrPollEvent(X.inst, &ev) == XR_SUCCESS) {
        if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
            const XrEventDataSessionStateChanged *sc = (const XrEventDataSessionStateChanged *)&ev;
            X.state = sc->state;
            fprintf(stderr, LOGX "session state %d\n", (int)X.state);
            if (X.state == XR_SESSION_STATE_READY) {
                XrSessionBeginInfo bi = {XR_TYPE_SESSION_BEGIN_INFO};
                bi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                if (XR_SUCCEEDED(xrBeginSession(X.session, &bi))) {
                    X.running = 1;
                    X.recenter = 1;
                    perf_and_refresh();
                }
            } else if (X.state == XR_SESSION_STATE_STOPPING) {
                if (X.running)
                    xrEndSession(X.session);
                X.running = 0;
            } else if (X.state == XR_SESSION_STATE_EXITING || X.state == XR_SESSION_STATE_LOSS_PENDING) {
                SDL_Event q;
                X.running = 0;
                memset(&q, 0, sizeof q);
                q.type = SDL_QUIT;
                SDL_PushEvent(&q);
            }
        } else if (ev.type == XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED) {
            log_profiles();
        }
        ev.type = XR_TYPE_EVENT_DATA_BUFFER;
        ev.next = NULL;
    }
}

static void end_frame(void)
{
    const XrCompositionLayerBaseHeader *layers[5];
    uint32_t nl = 0, idx[2] = {0, 0}, e, n = 0;
    XrCompositionLayerPassthroughFB ptl = {XR_TYPE_COMPOSITION_LAYER_PASSTHROUGH_FB};
    XrView views[2] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};
    XrCompositionLayerProjectionView pv[2] = {{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW},
                                              {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}};
    XrCompositionLayerProjection proj = {XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    XrCompositionLayerQuad ql = {XR_TYPE_COMPOSITION_LAYER_QUAD}, ml = {XR_TYPE_COMPOSITION_LAYER_QUAD};
    XrCompositionLayerSettingsFB qs = {XR_TYPE_COMPOSITION_LAYER_SETTINGS_FB};
    XrFrameEndInfo fei = {XR_TYPE_FRAME_END_INFO};
    int mr = X.pt_layer && !X.dark, show_menu = X.menu_open && X.menu_sc && X.fs.shouldRender;
    XrResult r;

    if (show_menu && X.menu_dirty) {
        uint32_t mi;
        if (acquire(X.menu_sc, &mi) && mi < X.menu_n) {
            VkCommandBuffer cb = own_begin();
            VkBufferImageCopy bc;
            draw_menu(X.menu_map);
            memset(&bc, 0, sizeof bc);
            bc.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            bc.imageSubresource.layerCount = 1;
            bc.imageExtent.width = MENU_W;
            bc.imageExtent.height = MENU_H;
            bc.imageExtent.depth = 1;
            barrier(cb, X.menu_img[mi], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            vkCmdCopyBufferToImage(cb, X.menu_buf, X.menu_img[mi], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bc);
            barrier(cb, X.menu_img[mi], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
            own_submit();
            nvk_queue_lock();
            release_locked(X.menu_sc);
            nvk_queue_unlock();
            X.menu_dirty = 0;
        }
    }
    if (mr) {
        ptl.layerHandle = X.pt_layer;
        ptl.flags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
        layers[nl++] = (const XrCompositionLayerBaseHeader *)&ptl;
    }
    if (X.fs.shouldRender) {
        XrViewLocateInfo vli = {XR_TYPE_VIEW_LOCATE_INFO};
        XrViewState vs = {XR_TYPE_VIEW_STATE};
        vli.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        vli.displayTime = X.fs.predictedDisplayTime;
        vli.space = X.local;
        if (XR_SUCCEEDED(xrLocateViews(X.session, &vli, &vs, 2, &n, views)) && n == 2
                && acquire(X.eye_sc[0], &idx[0])) {
            if (!acquire(X.eye_sc[1], &idx[1])) {
                nvk_queue_lock();
                release_locked(X.eye_sc[0]);
                nvk_queue_unlock();
            } else {
                VkCommandBuffer cb = own_begin();
                VkClearColorValue cc;
                VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                memset(&cc, 0, sizeof cc);
                cc.float32[3] = mr ? 0.0f : 1.0f;
                for (e = 0; e < 2; e++) {
                    VkImage im = X.eye_img[e][idx[e] < X.eye_n[e] ? idx[e] : 0];
                    barrier(cb, im, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT);
                    vkCmdClearColorImage(cb, im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cc, 1, &rr);
                    barrier(cb, im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
                }
                own_submit();
                nvk_queue_lock();
                release_locked(X.eye_sc[0]);
                release_locked(X.eye_sc[1]);
                nvk_queue_unlock();
                for (e = 0; e < 2; e++) {
                    pv[e].pose = views[e].pose;
                    pv[e].fov = views[e].fov;
                    pv[e].subImage.swapchain = X.eye_sc[e];
                    pv[e].subImage.imageRect.extent.width = PROJ_DIM;
                    pv[e].subImage.imageRect.extent.height = PROJ_DIM;
                }
                proj.layerFlags = mr ? XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT : 0;
                proj.space = X.local;
                proj.viewCount = 2;
                proj.views = pv;
                layers[nl++] = (const XrCompositionLayerBaseHeader *)&proj;
            }
        }
    }
    pthread_mutex_lock(&X.quad_mu);
    if (X.quad && X.quad_has_frame && X.fs.shouldRender) {
        if (X.ext_layer_settings) {
            qs.layerFlags = XR_COMPOSITION_LAYER_SETTINGS_QUALITY_SUPER_SAMPLING_BIT_FB
                          | XR_COMPOSITION_LAYER_SETTINGS_NORMAL_SHARPENING_BIT_FB;
            ql.next = &qs;
        }
        ql.space = X.local;
        ql.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
        ql.subImage.swapchain = X.quad;
        ql.subImage.imageRect.extent.width = (int32_t)X.quad_w;
        ql.subImage.imageRect.extent.height = (int32_t)X.quad_h;
        ql.pose = X.quad_pose;
        ql.size = X.quad_size;
        layers[nl++] = (const XrCompositionLayerBaseHeader *)&ql;
    }
    if (show_menu) {
        V3 nrm = qrot(X.quad_pose.orientation, v3(0, 0, 1));
        float mh = clampf(X.height * 0.75f, 0.45f, 2.4f);
        ml.space = X.local;
        ml.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
        ml.subImage.swapchain = X.menu_sc;
        ml.subImage.imageRect.extent.width = MENU_W;
        ml.subImage.imageRect.extent.height = MENU_H;
        ml.pose.orientation = X.quad_pose.orientation;
        ml.pose.position = vadd(X.quad_pose.position, vmul(nrm, 0.25f));
        ml.size.width = mh * (float)MENU_W / (float)MENU_H;
        ml.size.height = mh;
        layers[nl++] = (const XrCompositionLayerBaseHeader *)&ml;
    }
    fei.displayTime = X.fs.predictedDisplayTime;
    fei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    fei.layerCount = nl;
    fei.layers = layers;
    nvk_queue_lock();
    r = xrEndFrame(X.session, &fei);
    nvk_queue_unlock();
    pthread_mutex_unlock(&X.quad_mu);
    if (XR_FAILED(r)) {
        static int said;
        if (said++ < 5)
            fprintf(stderr, LOGX "xrEndFrame: %d\n", (int)r);
    }
}

static void *frame_main(void *arg)
{
    uint64_t frames = 0, t0 = now_ms();
    (void)arg;
    while (X.thread_run) {
        XrFrameWaitInfo fwi = {XR_TYPE_FRAME_WAIT_INFO};
        XrFrameBeginInfo fbi = {XR_TYPE_FRAME_BEGIN_INFO};
        XrResult r;
        poll_events();
        if (!X.running) {
            pthread_mutex_lock(&X.pad_mu);
            X.pad_active = 0;
            pthread_mutex_unlock(&X.pad_mu);
            usleep(20000);
            continue;
        }
        X.fs.type = XR_TYPE_FRAME_STATE;
        X.fs.next = NULL;
        if (XR_FAILED(xrWaitFrame(X.session, &fwi, &X.fs))) {
            usleep(5000);
            continue;
        }
        nvk_queue_lock();
        r = xrBeginFrame(X.session, &fbi);
        nvk_queue_unlock();
        if (XR_FAILED(r))
            continue;
        sync_actions();
        if (X.motor[0] != X.last_hap[0] || X.motor[1] != X.last_hap[1] || X.motor[0] || X.motor[1])
            refresh_haptics(X.motor[0] != X.last_hap[0] || X.motor[1] != X.last_hap[1]);
        if (X.recenter)
            do_recenter(X.fs.predictedDisplayTime);
        end_frame();
        if (++frames % 1200 == 0) {
            uint64_t t = now_ms();
            fprintf(stderr, LOGX "frame thread: %.1f frames/s, screen %ux%u, %s\n",
                    1200.0 * 1000.0 / (double)(t - t0 ? t - t0 : 1), X.quad_w, X.quad_h,
                    X.dark ? "dark room" : "mixed reality");
            t0 = t;
        }
    }
    return NULL;
}

int xrt_start(const struct nvk_core *C)
{
    XrGraphicsBindingVulkan2KHR gb = {XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR};
    XrSessionCreateInfo sci = {XR_TYPE_SESSION_CREATE_INFO};
    XrReferenceSpaceCreateInfo rs = {XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    VkCommandPoolCreateInfo cp = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    XrResult r;
    int e;

    if (!X.on || X.session)
        return X.on;
    X.C = C;
    pthread_mutex_init(&X.quad_mu, NULL);
    pthread_mutex_init(&X.pad_mu, NULL);
    gb.instance = C->inst;
    gb.physicalDevice = C->phys;
    gb.device = C->dev;
    gb.queueFamilyIndex = C->qfam;
    gb.queueIndex = 0;
    sci.next = &gb;
    sci.systemId = X.sys;
    r = xrCreateSession(X.inst, &sci, &X.session);
    if (XR_FAILED(r)) {
        fprintf(stderr, LOGX "xrCreateSession: %d\n", (int)r);
        X.on = 0;
        return 0;
    }
    rs.poseInReferenceSpace.orientation.w = 1.0f;
    rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    xrCreateReferenceSpace(X.session, &rs, &X.local);
    rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    xrCreateReferenceSpace(X.session, &rs, &X.view);
    X.quad_pose.orientation.w = 1.0f;
    X.quad_pose.position.z = -DEF_DIST;
    load_view();
    update_size();
    pick_formats();
    for (e = 0; e < 2; e++) {
        X.eye_sc[e] = make_sc(X.fmt_srgb, PROJ_DIM, PROJ_DIM,
                              XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT);
        X.eye_n[e] = X.eye_sc[e] ? enum_images(X.eye_sc[e], X.eye_img[e]) : 0;
    }
    cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cp.queueFamilyIndex = C->qfam;
    vkCreateCommandPool(C->dev, &cp, NULL, &X.pool);
    ai.commandPool = X.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    vkAllocateCommandBuffers(C->dev, &ai, &X.cb);
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(C->dev, &fi, NULL, &X.fence);
    if (!make_menu())
        fprintf(stderr, LOGX "no menu panel (the VR menu is off)\n");
    if (!make_actions())
        fprintf(stderr, LOGX "the Touch controllers could not be bound\n");
    if (X.ext_colorspace) {
        PFN_xrSetColorSpaceFB cs = (PFN_xrSetColorSpaceFB)xr_fn("xrSetColorSpaceFB");
        if (cs)
            cs(X.session, XR_COLOR_SPACE_REC709_FB);
    }
    init_passthrough();
    xbox_InputSetXrSource(xr_pad_state, xr_rumble);
    X.thread_run = 1;
    if (pthread_create(&X.thread, NULL, frame_main, NULL) != 0) {
        X.thread_run = 0;
        fprintf(stderr, LOGX "no frame thread\n");
        return 0;
    }
    fprintf(stderr, LOGX "session up on %s: eyes %s, passthrough %s, menu %s\n", C->name,
            X.eye_sc[0] ? "ok" : "MISSING", X.pt_layer ? "ok" : "none", X.menu_sc ? "ok" : "none");
    return 1;
}

void xrt_shutdown(void)
{
    if (!X.thread_run)
        return;
    X.thread_run = 0;
    pthread_join(X.thread, NULL);
    xbox_InputSetXrSource(NULL, NULL);
    if (X.session && X.running)
        xrRequestExitSession(X.session);
}

#else  /* not a headset build */

#include "xr_theater.h"
int  xrt_preinit(void) { return 0; }
int  xrt_on(void) { return 0; }
int  xrt_start(const struct nvk_core *C) { (void)C; return 0; }
void xrt_quad_record(VkCommandBuffer cb, VkImage src, VkFormat fmt, int x, int y, int w, int h)
{ (void)cb; (void)src; (void)fmt; (void)x; (void)y; (void)w; (void)h; }
void xrt_quad_submitted(int ok) { (void)ok; }
void xrt_shutdown(void) {}

#endif
