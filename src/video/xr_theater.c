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
#include "d3d/nv2a_vk_glsl.h"
#include "input/xinput_xbox.h"
#include "input/touch_pad.h"
#include "xr_theater.h"

unsigned nv2a_host_render_scale_x2(void);
static void perf_and_refresh(void);

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

/* Defaults: Futurama Quest's placement, at 16:9. */
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

enum { M_RESUME, M_MODE, M_LEVEL, M_FILL, M_BACKGROUND, M_3D, M_RECENTER, M_RESET, M_EXIT, M_COUNT };
static const char *const k_fill_name[3] = {"16:9", "4:3", "FULL"};
static const float k_fill_aspect[3] = {0.0f, 0.75f, 1.0f};

/* 3D depth levels: the shift at infinity, in picture widths (Futurama's
 * 0.022 clip-space units x its 1 / 1.6 / 2.3 scale, halved for uv). */
static const float k_depth[4] = {0.0f, 0.011f, 0.0176f, 0.0253f};
static const char *const k_depth_name[4] = {"OFF", "NORMAL", "DEEP", "EXTRA"};

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
    VkImageView quad_view[MAXIMG];
    /* the draw into it (GT3-revisited's lesson: a render pass, not a
     * transfer copy, into a Quest swapchain image) */
    int draw_ok;                 /* -1 failed: the copy instead */
    int depth_level, last_depth_on, prev_3d_chord;
    int quad_stereo;             /* the quad is side by side, an eye a half */
    int sdraw_ok;                /* the 3D pipeline: 0 not yet, 1 ok, -1 failed */
    VkPipelineLayout spl;
    VkPipeline spipe;
    VkDescriptorSetLayout sdsl;
    VkDescriptorPool sdpool;
    VkSampler snear;
    struct { VkImage img; VkImageView fv, sv, dv; VkDescriptorSet set; } ssrc[16];
    int nssrc;
    uint64_t frames3d, frames2d;
    /* head-tracked mode (MODE: IMMERSIVE) */
    int immersive;
    float hz_live;               /* the refresh the menu's mode switch asked for */
    pthread_mutex_t pose_mu;
    XrView lv[2];                /* the latest located views (predicted display time) */
    XrPosef lhead;               /* and the head (VIEW space in LOCAL) */
    int lvalid;
    XrVector3f org_pos;          /* the re-centred origin: the head's place and heading */
    XrQuaternionf org_q;         /* heading only */
    int org_valid;
    struct { XrView v[2]; float tanh, tanv; } pring[64];
    int level, fill;             /* level horizon; the picture: 0 the game's, 1 4:3, 2 full */
    XrSwapchain hud_sc;          /* the HUD's own panel (head-tracked mode) */
    VkImage hud_img[MAXIMG];
    VkImageView hud_view[MAXIMG];
    uint32_t hud_n, hud_idx;
    int hud_acquired, hud_has, hud_pending;
    VkPipeline hpipe;
    uint32_t pose_id;            /* the last index handed out (0 = none yet) */
    uint32_t quad_pose_pending, quad_pose_id;   /* the quad's picture's pose index */
    uint64_t imm_frames, imm_noposes;
    VkPipelineLayout pl;
    VkPipeline pipe;
    VkDescriptorSetLayout dsl;
    VkDescriptorPool dpool;
    VkSampler smp;
    struct { VkImage img; VkImageView view; VkDescriptorSet set; } src[MAXIMG];
    int nsrc;
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

/* The runtime makes them through this: vkCreateInstance and vkCreateDevice
 * are the toolkit's own (d3d/vk_dyn.c fills its function pointers in them,
 * from the new instance and device), everything else the loader's. */
static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL xr_gipa(VkInstance inst, const char *name)
{
    if (!strcmp(name, "vkCreateInstance"))
        return (PFN_vkVoidFunction)vkCreateInstance;
    if (!strcmp(name, "vkCreateDevice") && inst)
        return (PFN_vkVoidFunction)vkCreateDevice;
    return vkGetInstanceProcAddr(inst, name);
}
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
    xi.pfnGetInstanceProcAddr = xr_gipa;
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
    xd.pfnGetInstanceProcAddr = xr_gipa;
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
    xbox_InputTouchForceOff(1);   /* no on-screen controls in the headset */
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
    X.depth_level = 1;
    X.level = 1;                 /* level horizon on by default: steadier */
    if (!dir)
        return;
    snprintf(X.prefs, sizeof X.prefs, "%s/xr_view.txt", dir);
    f = fopen(X.prefs, "r");
    if (!f)
        return;
    X.depth_level = 1;
    if (fscanf(f, "%7s %f %f %f %d", tag, &d, &h, &e, &dk) == 5
            && (!strcmp(tag, "t1") || !strcmp(tag, "t2") || !strcmp(tag, "t3") || !strcmp(tag, "t4"))) {
        int lv = 1, im = 0, le = 1, fi = 0;
        if (strcmp(tag, "t1") && fscanf(f, "%d", &lv) == 1)
            X.depth_level = lv < 0 ? 0 : lv > 3 ? 3 : lv;
        if ((!strcmp(tag, "t3") || !strcmp(tag, "t4")) && fscanf(f, "%d", &im) == 1)
            X.immersive = im != 0;
        if (!strcmp(tag, "t4") && fscanf(f, "%d %d", &le, &fi) == 2) {
            X.level = le != 0;
            X.fill = fi < 0 ? 0 : fi > 2 ? 2 : fi;
        }
        X.dist = clampf(d, MIN_DIST, MAX_DIST);
        X.height = clampf(h, MIN_H, MAX_H);
        X.elev = clampf(e, -1.2f, 1.2f);
        X.dark = dk != 0;
        fprintf(stderr, LOGX "view restored: %.2f m away, %.2f m tall, elevation %.2f, %s, 3D %s\n", X.dist,
                X.height, X.elev, X.dark ? "dark room" : "mixed reality", k_depth_name[X.depth_level]);
    }
    fclose(f);
}

static void save_view(void)
{
    FILE *f;
    X.view_dirty = 0;
    if (!X.prefs[0] || !(f = fopen(X.prefs, "w")))
        return;
    fprintf(f, "t4 %.3f %.3f %.3f %d %d %d %d %d\n", X.dist, X.height, X.elev, X.dark, X.depth_level, X.immersive,
            X.level, X.fill);
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
    {   /* the head-tracked mode's origin: here, facing this way (heading only) */
        float yaw = atan2f(-fx, -fz);
        pthread_mutex_lock(&X.pose_mu);
        X.org_pos = loc.pose.position;
        X.org_q.x = 0.0f;
        X.org_q.y = sinf(yaw * 0.5f);
        X.org_q.z = 0.0f;
        X.org_q.w = cosf(yaw * 0.5f);
        X.org_valid = 1;
        pthread_mutex_unlock(&X.pose_mu);
    }
    X.recenter = 0;
}

/* ---- head-tracked mode: the poses the game draws with ---- */
static XrQuaternionf qmul(XrQuaternionf a, XrQuaternionf b)
{
    XrQuaternionf r;
    r.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    r.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    r.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
    r.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
    return r;
}
static XrQuaternionf qconj(XrQuaternionf q) { q.x = -q.x; q.y = -q.y; q.z = -q.z; return q; }

int xrt_immersive(void) { return X.on && X.running && X.immersive && X.org_valid && X.lvalid; }

void xrt_immersive_opts(int *level, float *aspect)
{
    *level = X.level;
    *aspect = k_fill_aspect[X.fill];
}

void xrt_pose_set_tanv(uint32_t id, float tanv)
{
    pthread_mutex_lock(&X.pose_mu);
    if (id && X.pring[id % 64].tanh > 0.0f)
        X.pring[id % 64].tanv = tanv;
    pthread_mutex_unlock(&X.pose_mu);
}

uint32_t xrt_pose_take(float q[4], float p[3], float *tan_half_h)
{
    XrQuaternionf r, c0;
    V3 d;
    uint32_t id;
    float t = 0.0f;
    int e;
    if (!xrt_immersive())
        return 0;
    pthread_mutex_lock(&X.pose_mu);
    id = ++X.pose_id;
    if (!id)
        id = ++X.pose_id;
    X.pring[id % 64].v[0] = X.lv[0];
    X.pring[id % 64].v[1] = X.lv[1];
    for (e = 0; e < 2; e++) {
        float a = tanf(X.lv[e].fov.angleRight), b = tanf(-X.lv[e].fov.angleLeft);
        t = fmaxf(t, fmaxf(a, b));
    }
    X.pring[id % 64].tanh = t;
    X.pring[id % 64].tanv = 0.0f;   /* the game says (xrt_pose_set_tanv) */
    c0 = qconj(X.org_q);
    r = qmul(c0, X.lhead.orientation);
    d = qrot(c0, vsub(X.lhead.position, X.org_pos));
    pthread_mutex_unlock(&X.pose_mu);
    q[0] = r.x; q[1] = r.y; q[2] = r.z; q[3] = r.w;
    p[0] = d.x; p[1] = d.y; p[2] = d.z;
    *tan_half_h = t;
    return id;
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
    static const char *labels[M_COUNT] = {"RESUME", "MODE", "LEVEL HORIZON", "VIEW", "BACKGROUND", "3D DEPTH",
                                          "RECENTER", "RESET SCREEN", "EXIT GAME"};
    const char *values[M_COUNT] = {"", !X.immersive ? "SCREEN" : nv2a_host_render_scale_x2() < 4 ? "IMMERSIVE: RESTART FOR HD" : "IMMERSIVE (BETA)",
                                   X.level ? "ON" : "OFF", k_fill_name[X.fill],
                                   X.dark || !X.pt_layer ? "DARK ROOM" : "MIXED REALITY",
                                   k_depth_name[X.depth_level], "", "", X.exit_armed ? "PRESS A AGAIN" : ""};
    const int band = 150, top = band + 6 + 26;
    int i;
    mfill(img, 0, 0, MENU_W, MENU_H, 0xF26B1DFFu);              /* orange border */
    mfill(img, 6, 6, MENU_W - 12, MENU_H - 12, 0x101014FFu);     /* dark panel */
    mfill(img, 6, 6, MENU_W - 12, band, 0x2A0F08FFu);            /* header band */
    mtext(img, (MENU_W - mwidth("BURNOUT 3", 12)) / 2, 6 + 16, 12, 0xF26B1DFFu, "BURNOUT 3");
    mtext(img, (MENU_W - mwidth("TAKEDAWN VR", 4)) / 2, 6 + 16 + 90, 4, 0xFFD24AFFu, "TAKEDAWN VR");
    for (i = 0; i < M_COUNT; i++) {
        int y = top + i * 50, sel = i == X.menu_sel;
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
    case M_MODE:
        X.immersive = !X.immersive;
        nvk_xr_scene_wanted = X.depth_level > 0 || X.immersive;
        X.recenter = 1;
        fprintf(stderr, LOGX "mode: %s\n", X.immersive ? "immersive (head-tracked)" : "screen");
        /* immersive at 90 Hz, the screen at 120 (the start does the same,
         * GameActivity); RECOMP_XR_HZ in env.txt keeps its own */
        if (!getenv("RECOMP_XR_HZ") || X.hz_live > 0.0f) {
            X.hz_live = X.immersive ? 90.0f : 120.0f;
            perf_and_refresh();
        }
        save_view();
        break;
    case M_LEVEL:
        X.level = !X.level;
        save_view();
        break;
    case M_FILL:
        X.fill = (X.fill + (dir < 0 ? 2 : 1)) % 3;
        save_view();
        break;
    case M_BACKGROUND:
        if (!X.pt_layer)
            break;
        X.dark = !X.dark;
        apply_background();
        save_view();
        break;
    case M_3D:
        X.depth_level = (X.depth_level + (dir < 0 ? 3 : 1)) % 4;
        if (X.depth_level)
            X.last_depth_on = X.depth_level;
        nvk_xr_scene_wanted = X.depth_level > 0 || X.immersive;
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
    {   /* LB + RB + L3: 3D on / off (the last level) */
        int c3 = window && l3 && !r3;
        if (c3 && !X.prev_3d_chord) {
            if (X.depth_level)
                X.last_depth_on = X.depth_level, X.depth_level = 0;
            else
                X.depth_level = X.last_depth_on ? X.last_depth_on : 1;
            nvk_xr_scene_wanted = X.depth_level > 0 || X.immersive;
            X.view_dirty = 1;
            X.menu_dirty = 1;
            fprintf(stderr, LOGX "3D %s\n", k_depth_name[X.depth_level]);
        }
        X.prev_3d_chord = c3;
    }
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
        if (X.hz_live > 0.0f)
            w = X.hz_live;   /* the mode switched in the menu */
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

/* ---- the draw: the presenter's image, sampled and scaled into the quad ---- */
static const char k_vs[] =
    "#version 450\n"
    "layout(location = 0) out vec2 uv;\n"
    "void main() {\n"
    "    uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);\n"
    "    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);\n"
    "}\n";
/* The bytes are sRGB-encoded in a UNORM image: decoded here, encoded again
 * by the sRGB attachment, so the filtering is the only change. */
static const char k_fs[] =
    "#version 450\n"
    "layout(set = 0, binding = 0) uniform sampler2D t;\n"
    "layout(push_constant) uniform P { vec4 r; } p;\n"
    "layout(location = 0) in vec2 uv;\n"
    "layout(location = 0) out vec4 o;\n"
    "vec3 lin(vec3 c) {\n"
    "    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));\n"
    "}\n"
    "void main() {\n"
    "    o = vec4(lin(texture(t, p.r.xy + uv * p.r.zw).rgb), 1.0);\n"
    "}\n";

static VkShaderModule shader(enum nvk_stage st, const char *src)
{
    const uint32_t *w;
    size_t n;
    int compiled;
    char err[1024];
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    VkShaderModule m = VK_NULL_HANDLE;
    if (!nvk_glsl_spirv(st, src, &w, &n, &compiled, err, sizeof err)) {
        fprintf(stderr, LOGX "shader: %s\n", err);
        return VK_NULL_HANDLE;
    }
    ci.codeSize = n * 4;
    ci.pCode = w;
    vkCreateShaderModule(X.C->dev, &ci, NULL, &m);
    return m;
}

static int draw_init(VkFormat fmt)
{
    VkPipelineRenderingCreateInfo pr = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    VkDescriptorSetLayoutBinding b = {0};
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    VkPushConstantRange pc = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAXIMG};
    VkDescriptorPoolCreateInfo dp = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    VkSamplerCreateInfo sci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    VkShaderModule vs, fs;
    VkPipelineShaderStageCreateInfo st[2];
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba = {0};
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    VkGraphicsPipelineCreateInfo gp = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    VkResult r;

    pr.colorAttachmentCount = 1;
    pr.pColorAttachmentFormats = &fmt;
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    dl.bindingCount = 1;
    dl.pBindings = &b;
    if (vkCreateDescriptorSetLayout(X.C->dev, &dl, NULL, &X.dsl) != VK_SUCCESS)
        return 0;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &X.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(X.C->dev, &plci, NULL, &X.pl) != VK_SUCCESS)
        return 0;
    dp.maxSets = MAXIMG;
    dp.poolSizeCount = 1;
    dp.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(X.C->dev, &dp, NULL, &X.dpool) != VK_SUCCESS)
        return 0;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = 0.0f;
    if (vkCreateSampler(X.C->dev, &sci, NULL, &X.smp) != VK_SUCCESS)
        return 0;
    vs = shader(NVK_STAGE_VERTEX, k_vs);
    fs = shader(NVK_STAGE_FRAGMENT, k_fs);
    if (!vs || !fs)
        return 0;
    memset(st, 0, sizeof st);
    st[0].sType = st[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    st[0].module = vs;
    st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    st[1].module = fs;
    st[1].pName = "main";
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.lineWidth = 1.0f;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    cba.colorWriteMask = 0xF;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;
    ds.dynamicStateCount = 2;
    ds.pDynamicStates = dyn;
    gp.stageCount = 2;
    gp.pStages = st;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pColorBlendState = &cb;
    gp.pDynamicState = &ds;
    gp.pNext = &pr;      /* dynamic rendering, as the rest of the presenter */
    gp.layout = X.pl;
    r = vkCreateGraphicsPipelines(X.C->dev, VK_NULL_HANDLE, 1, &gp, NULL, &X.pipe);
    vkDestroyShaderModule(X.C->dev, vs, NULL);
    vkDestroyShaderModule(X.C->dev, fs, NULL);
    return r == VK_SUCCESS;
}

static VkImageView view_of(VkImage img, VkFormat fmt)
{
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    VkImageView v = VK_NULL_HANDLE;
    vi.image = img;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = fmt;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    vkCreateImageView(X.C->dev, &vi, NULL, &v);
    return v;
}

/* The presenter's swapchain images, each with a view and a descriptor set,
 * made at first sight (the swapchain is made again on a resize: new images,
 * the old entries are dropped then). */
static VkDescriptorSet src_set(VkImage img, VkFormat fmt)
{
    int i;
    VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    VkDescriptorImageInfo ii;
    VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    for (i = 0; i < X.nsrc; i++)
        if (X.src[i].img == img)
            return X.src[i].set;
    if (X.nsrc == MAXIMG) {     /* a new swapchain: start over */
        vkResetDescriptorPool(X.C->dev, X.dpool, 0);
        for (i = 0; i < X.nsrc; i++)
            vkDestroyImageView(X.C->dev, X.src[i].view, NULL);
        X.nsrc = 0;
    }
    i = X.nsrc;
    X.src[i].img = img;
    X.src[i].view = view_of(img, fmt);
    ai.descriptorPool = X.dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &X.dsl;
    if (!X.src[i].view || vkAllocateDescriptorSets(X.C->dev, &ai, &X.src[i].set) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    ii.sampler = X.smp;
    ii.imageView = X.src[i].view;
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    w.dstSet = X.src[i].set;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &ii;
    vkUpdateDescriptorSets(X.C->dev, 1, &w, 0, NULL);
    X.nsrc++;
    return X.src[i].set;
}

/* ---- the 3D: each eye from the scene and its depth ----
 * The Futurama shift, x += eye * S * (1 - focus / w), is linear in the depth
 * buffer's value (window z is linear in 1/w), so per pixel the shift is
 * S * (d - f) / (1 - f): 0 at the focus depth f, S at the far plane (the sky),
 * negative (out of the screen) nearer than f, held to -pop. f is taken each
 * frame from three points where the car sits in the chase view. An eye's
 * pixel is looked up where its shift brings it from (a fixed point, four
 * steps). The HUD (the frame's pixels the scene lacks: frame != scene) is
 * not shifted: it stays on the screen plane, flat, as in Futurama. */
static const char k_sfs[] =
    "#version 450\n"
    "layout(set = 0, binding = 0) uniform sampler2D fin;\n"   /* the presented picture (post passes) */
    "layout(set = 0, binding = 1) uniform sampler2D frm;\n"   /* the frame with its HUD */
    "layout(set = 0, binding = 2) uniform sampler2D scn;\n"   /* the scene before the HUD */
    "layout(set = 0, binding = 3) uniform sampler2D dep;\n"   /* its depth */
    /* ph: near, far, 1 = the head-tracked mode's physical shift (a * 1/z,
     * 0 at infinity, st.y the nearest 1/z), 0 = the screen's (focus at the car) */
    "layout(push_constant) uniform P { vec4 rf; vec4 rs; vec4 st; vec4 ph; } p;\n"
    "layout(location = 0) in vec2 uv;\n"
    "layout(location = 0) out vec4 o;\n"
    "vec3 lin(vec3 c) {\n"
    "    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));\n"
    "}\n"
    "vec2 S(vec2 u) { return p.rs.xy + clamp(u, 0.0, 1.0) * p.rs.zw; }\n"
    /* the nearest of three texels across: foreground edges a little wider,
     * so the background does not show through as slivers beside thin
     * near things (fence posts) */
    "float D(vec2 u) {\n"
    "    vec2 t = S(u);\n"
    "    float dx = 1.5 / float(textureSize(dep, 0).x);\n"
    "    return min(texture(dep, t).r, min(texture(dep, t + vec2(dx, 0.0)).r, texture(dep, t - vec2(dx, 0.0)).r));\n"
    "}\n"
    "bool hud(vec2 u) {\n"
    "    vec3 d = abs(texture(frm, S(u)).rgb - texture(scn, S(u)).rgb);\n"
    "    return max(d.r, max(d.g, d.b)) > p.st.z;\n"
    "}\n"
    "float D1(vec2 u) { return texture(dep, S(u)).r; }\n"
    "void main() {\n"
    "    float f = (D1(vec2(0.5, 0.62)) + D1(vec2(0.42, 0.68)) + D1(vec2(0.58, 0.68))) / 3.0;\n"
    "    f = min(f, 0.99999);\n"
    "    float a = p.st.x, pop = p.st.y;\n"
    "    bool phys = p.ph.z > 0.5;\n"
    "    float kmin = phys ? 0.0 : -pop, kmax = phys ? pop : 1.0;\n"
    /* the inverse warp: the source pixels whose shifted place is this one,
     * the nearest of them (the front surface); none is a hole (background
     * uncovered), filled by the fixed-point lookup from the far side */
    "    float sh0 = a * kmax, sh1 = a * kmin;\n"
    "    float lo = uv.x - max(sh0, sh1), hi = uv.x - min(sh0, sh1);\n"
    /* a step of at most one source texel: coarser steps (24 over the
     * range at DEEP and EXTRA) let the car's bright edge pixels land up to
     * half a step outside its outline, as specks along its sides */
    "    float tw = float(textureSize(dep, 0).x) * p.rs.z;\n"
    "    int N = clamp(int(ceil((hi - lo) * tw)), 8, 96);\n"
    "    float step_ = max((hi - lo) / float(N), 1e-6);\n"
    "    float best = -1.0, bestd = 2.0;\n"
    /* the farthest surface in the range and, of its samples, the one whose
     * shifted place is nearest: what a hole (background the shift uncovers
     * beside a near thing) is stretched from, never the near thing itself
     * (dark wedges off the car's wings over a bright road) */
    "    float bg = -1.0, bgd = -1.0, bge = 1e9;\n"
    "    float thr = phys ? 2e-6 : 0.0012 * max(1.0 - f, 1e-4);\n"
    /* away from depth edges one shift serves the whole range: the cheap
     * fixed-point lookup; the search only where the range crosses an edge */
    /* nine samples across the range (two texels apart at most at NORMAL):
     * three let thin near things between them (the car's side mirrors)
     * take the cheap path and come apart from the car */
    "    float emn = 2.0, emx = -1.0;\n"
    "    for (int j = 0; j < 9; j++) { float e = D1(vec2(mix(lo, hi, float(j) / 8.0), uv.y)); emn = min(emn, e); emx = max(emx, e); }\n"
    "    bool edge = emx - emn > (phys ? 2e-6 : 0.0012 * max(1.0 - f, 1e-4));\n"
    "    for (int j = 0; j < N && edge; j++) {\n"
    "        float s = lo + (float(j) + 0.5) * step_;\n"
    /* the widened depth: a soft edge's pixels (the post's colour over
     * the background's depth) go with the post, not with what is behind */
    "        float d = D(vec2(s, uv.y));\n"
    "        float k = phys ? clamp((1.0 - d * (p.ph.y - p.ph.x) / p.ph.y) / p.ph.x, kmin, kmax)\n"
    "                       : clamp((d - f) / max(1.0 - f, 1e-6), kmin, kmax);\n"
    "        float er = abs(s + a * k - uv.x);\n"
    "        if (er <= step_ * 0.75 && d < bestd) { bestd = d; best = s; }\n"
    "        float d1 = D1(vec2(s, uv.y));\n"
    "        if (d1 > bgd + thr) { bgd = d1; bg = s; bge = er; }\n"
    "        else if (d1 > bgd - thr && er < bge) { bg = s; bge = er; if (d1 > bgd) bgd = d1; }\n"
    "    }\n"
    "    float s = best;\n"
    "    if (s < 0.0 && edge && bg >= 0.0) s = bg;\n"
    "    if (s < 0.0) {\n"
    "        s = uv.x;\n"
    "        for (int i = 0; i < 4; i++) {\n"
    "            float dd = edge ? D(vec2(s, uv.y)) : D1(vec2(s, uv.y));\n"
    "            float k = phys ? clamp((1.0 - dd * (p.ph.y - p.ph.x) / p.ph.y) / p.ph.x, kmin, kmax)\n"
    "                           : clamp((dd - f) / max(1.0 - f, 1e-6), kmin, kmax);\n"
    "            s = uv.x - a * k;\n"
    "        }\n"
    "    }\n"
    "    vec2 su = vec2(clamp(s, 0.0, 1.0), uv.y);\n"
    "    vec3 c;\n"
    /* head-tracked: the HUD is on its own panel, never in the eyes */
    /* screen: the HUD and menus over a 3D scene at the nearest depth the
     * scene can come out to (k = -pop), so nothing of the scene comes
     * through in front of them (a results page over the race) */
    "    vec2 hu = vec2(uv.x + a * pop, uv.y);\n"
    "    bool hh = !phys && hud(hu);\n"
    "    if (phys) c = hud(su) ? texture(scn, S(su)).rgb : texture(fin, p.rf.xy + su * p.rf.zw).rgb;\n"
    "    else if (hh) c = texture(fin, p.rf.xy + hu * p.rf.zw).rgb;\n"
    "    else if (hud(su)) c = texture(scn, S(su)).rgb;\n"
    "    else c = texture(fin, p.rf.xy + su * p.rf.zw).rgb;\n"
    /* RECOMP_XR_DEBUG=mask: what the 3D takes for HUD, tinted red */
    "    if (p.ph.w > 0.5 && p.ph.w < 1.5 && (hh || hud(su))) c = mix(c, vec3(1.0, 0.0, 0.0), 0.6);\n"
    /* RECOMP_XR_DEBUG=depth: the depth the 3D uses (white = the focus,
     * dark = far), the HUD in red */
    "    if (p.ph.w > 1.5) { float dv = D1(uv); float kk = clamp((dv - f) / max(1.0 - f, 1e-6), 0.0, 1.0);\n"
    "        c = hud(uv) ? vec3(1.0, 0.0, 0.0) : vec3(pow(1.0 - kk, 3.0)); }\n"
    "    o = vec4(lin(c), 1.0);\n"
    "}\n";

/* The head-tracked mode's HUD panel: the frame's pixels the scene lacks,
 * the rest transparent. */
static const char k_hfs[] =
    "#version 450\n"
    "layout(set = 0, binding = 1) uniform sampler2D frm;\n"
    "layout(set = 0, binding = 2) uniform sampler2D scn;\n"
    "layout(push_constant) uniform P { vec4 rf; vec4 rs; vec4 st; vec4 ph; } p;\n"
    "layout(location = 0) in vec2 uv;\n"
    "layout(location = 0) out vec4 o;\n"
    "vec3 lin(vec3 c) {\n"
    "    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));\n"
    "}\n"
    "void main() {\n"
    "    vec2 t = p.rs.xy + uv * p.rs.zw;\n"
    "    vec3 a = texture(frm, t).rgb, d = abs(a - texture(scn, t).rgb);\n"
    "    o = max(d.r, max(d.g, d.b)) > p.st.z ? vec4(lin(a), 1.0) : vec4(0.0);\n"
    "}\n";

static int sdraw_init(VkFormat fmt)
{
    VkPipelineRenderingCreateInfo pr = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    VkDescriptorSetLayoutBinding b[4];
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    VkPushConstantRange pc = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 64};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16 * 4};
    VkDescriptorPoolCreateInfo dp = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    VkSamplerCreateInfo sci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    VkShaderModule vs, fs;
    VkPipelineShaderStageCreateInfo st[2];
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba = {0};
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    VkGraphicsPipelineCreateInfo gp = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    VkResult r;
    int i;

    pr.colorAttachmentCount = 1;
    pr.pColorAttachmentFormats = &fmt;
    memset(b, 0, sizeof b);
    for (i = 0; i < 4; i++) {
        b[i].binding = (uint32_t)i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    dl.bindingCount = 4;
    dl.pBindings = b;
    if (vkCreateDescriptorSetLayout(X.C->dev, &dl, NULL, &X.sdsl) != VK_SUCCESS)
        return 0;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &X.sdsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(X.C->dev, &plci, NULL, &X.spl) != VK_SUCCESS)
        return 0;
    dp.maxSets = 16;
    dp.poolSizeCount = 1;
    dp.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(X.C->dev, &dp, NULL, &X.sdpool) != VK_SUCCESS)
        return 0;
    sci.magFilter = VK_FILTER_NEAREST;   /* the frame, scene and depth: texel for texel */
    sci.minFilter = VK_FILTER_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(X.C->dev, &sci, NULL, &X.snear) != VK_SUCCESS)
        return 0;
    vs = shader(NVK_STAGE_VERTEX, k_vs);
    fs = shader(NVK_STAGE_FRAGMENT, k_sfs);
    if (!vs || !fs)
        return 0;
    memset(st, 0, sizeof st);
    st[0].sType = st[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    st[0].module = vs;
    st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    st[1].module = fs;
    st[1].pName = "main";
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.lineWidth = 1.0f;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    cba.colorWriteMask = 0xF;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;
    ds.dynamicStateCount = 2;
    ds.pDynamicStates = dyn;
    gp.pNext = &pr;
    gp.stageCount = 2;
    gp.pStages = st;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pColorBlendState = &cb;
    gp.pDynamicState = &ds;
    gp.layout = X.spl;
    r = vkCreateGraphicsPipelines(X.C->dev, VK_NULL_HANDLE, 1, &gp, NULL, &X.spipe);
    vkDestroyShaderModule(X.C->dev, fs, NULL);
    if (r == VK_SUCCESS) {   /* the HUD panel's: same layout and set */
        VkShaderModule hs = shader(NVK_STAGE_FRAGMENT, k_hfs);
        if (hs) {
            st[1].module = hs;
            if (vkCreateGraphicsPipelines(X.C->dev, VK_NULL_HANDLE, 1, &gp, NULL, &X.hpipe) != VK_SUCCESS)
                X.hpipe = VK_NULL_HANDLE;
            vkDestroyShaderModule(X.C->dev, hs, NULL);
        }
    }
    vkDestroyShaderModule(X.C->dev, vs, NULL);
    return r == VK_SUCCESS;
}

/* A set per (presented image, slot): their views do not change while the
 * images live; a full table starts over. */
static VkDescriptorSet ssrc_set(VkImage src, VkFormat fmt, const struct xrt_scene *s)
{
    VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    VkDescriptorImageInfo ii[4];
    VkWriteDescriptorSet w[4];
    VkImageView fv;
    int i;
    for (i = 0; i < X.nssrc; i++)
        if (X.ssrc[i].img == src && X.ssrc[i].sv == s->scene_view && X.ssrc[i].dv == s->depth_view)
            return X.ssrc[i].set;
    if (X.nssrc == 16) {
        vkResetDescriptorPool(X.C->dev, X.sdpool, 0);
        X.nssrc = 0;
    }
    /* the presented image's view: the one draw path's (src_set made it) */
    if (!src_set(src, fmt))
        return VK_NULL_HANDLE;
    fv = VK_NULL_HANDLE;
    for (i = 0; i < X.nsrc; i++)
        if (X.src[i].img == src)
            fv = X.src[i].view;
    i = X.nssrc;
    ai.descriptorPool = X.sdpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &X.sdsl;
    if (!fv || vkAllocateDescriptorSets(X.C->dev, &ai, &X.ssrc[i].set) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    X.ssrc[i].img = src;
    X.ssrc[i].fv = s->frame_view;
    X.ssrc[i].sv = s->scene_view;
    X.ssrc[i].dv = s->depth_view;
    ii[0].sampler = X.smp;
    ii[0].imageView = fv;
    ii[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ii[1].sampler = X.snear;
    ii[1].imageView = s->frame_view;
    ii[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ii[2] = ii[1];
    ii[2].imageView = s->scene_view;
    ii[3] = ii[1];
    ii[3].imageView = s->depth_view;
    for (int k = 0; k < 4; k++) {
        memset(&w[k], 0, sizeof w[k]);
        w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[k].dstSet = X.ssrc[i].set;
        w[k].dstBinding = (uint32_t)k;
        w[k].descriptorCount = 1;
        w[k].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[k].pImageInfo = &ii[k];
    }
    vkUpdateDescriptorSets(X.C->dev, 4, w, 0, NULL);
    X.nssrc++;
    return X.ssrc[i].set;
}

/* The head-tracked mode's HUD panel: 1280x720, its own swapchain. */
#define HUD_W 1280
#define HUD_H 720
static void hud_pass(VkCommandBuffer cb, VkDescriptorSet set, const struct xrt_scene *s)
{
    VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    VkRenderingInfo rb = {VK_STRUCTURE_TYPE_RENDERING_INFO};
    VkViewport v = {0, 0, (float)HUD_W, (float)HUD_H, 0, 1};
    VkRect2D sc = {{0, 0}, {HUD_W, HUD_H}};
    float pc[16];
    uint32_t i;
    if (!X.hud_sc) {
        X.hud_sc = make_sc(X.fmt_srgb, HUD_W, HUD_H, XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT
                                                     | XR_SWAPCHAIN_USAGE_SAMPLED_BIT);
        X.hud_n = X.hud_sc ? enum_images(X.hud_sc, X.hud_img) : 0;
        for (i = 0; i < X.hud_n; i++)
            X.hud_view[i] = view_of(X.hud_img[i], (VkFormat)X.fmt_srgb);
        fprintf(stderr, LOGX "HUD panel swapchain %dx%d (%s)\n", HUD_W, HUD_H, X.hud_sc ? "made" : "FAILED");
    }
    if (!X.hud_sc || X.hud_acquired || !acquire(X.hud_sc, &X.hud_idx) || X.hud_idx >= X.hud_n)
        return;
    X.hud_acquired = 1;
    memset(pc, 0, sizeof pc);
    pc[4] = (float)s->rect[0] / (float)s->w;
    pc[5] = (float)s->rect[1] / (float)s->h;
    pc[6] = (float)s->rect[2] / (float)s->w;
    pc[7] = (float)s->rect[3] / (float)s->h;
    pc[10] = 3.0f / 255.0f;
    barrier(cb, X.hud_img[X.hud_idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    ca.imageView = X.hud_view[X.hud_idx];
    ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    rb.renderArea = sc;
    rb.layerCount = 1;
    rb.colorAttachmentCount = 1;
    rb.pColorAttachments = &ca;
    vkCmdBeginRendering(cb, &rb);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, X.hpipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, X.spl, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(cb, X.spl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 64, pc);
    vkCmdSetViewport(cb, 0, 1, &v);
    vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
}

/* ---- the presenter's side ---- */
/* The screen's size: the picture's, at most XR_MAX_W wide (the immersive
 * window is the panel's 4128x2208; the game draws a fraction of that). */
#define XR_MAX_W 1920

void xrt_quad_record(VkCommandBuffer cb, VkImage src, VkFormat fmt, int x, int y, int w, int h,
                     int sampled, uint32_t sw, uint32_t sh, const struct xrt_scene *scene)
{
    VkImageCopy c;
    int64_t want;
    uint32_t qw, qh, i, scw;
    int draw, stereo;
    if (!X.on || !X.session || w <= 0 || h <= 0 || X.quad_acquired)
        return;
    if (X.draw_ok == 0 && sampled)
        X.draw_ok = draw_init((VkFormat)X.fmt_srgb) ? 1 : -1;
    draw = sampled && X.draw_ok > 0;
    if (draw && (X.depth_level > 0 || X.immersive) && X.sdraw_ok == 0)
        X.sdraw_ok = sdraw_init((VkFormat)X.fmt_srgb) ? 1 : -1;
    stereo = draw && (X.depth_level > 0 || X.immersive) && X.sdraw_ok > 0;
    if (draw) {
        want = X.fmt_srgb;
        qw = (uint32_t)w > XR_MAX_W ? XR_MAX_W : (uint32_t)w;
        qh = (uint32_t)(((uint64_t)h * qw + (uint32_t)w / 2) / (uint32_t)w);
    } else {
        want = (fmt == VK_FORMAT_B8G8R8A8_UNORM || fmt == VK_FORMAT_B8G8R8A8_SRGB) ? X.fmt_srgb_bgra : X.fmt_srgb;
        qw = (uint32_t)w;
        qh = (uint32_t)h;
    }
    if (!want)
        return;      /* no matching format: a raw copy would swap the colours */
    pthread_mutex_lock(&X.quad_mu);
    scw = stereo ? qw * 2 : qw;
    if (!X.quad || X.quad_w != qw || X.quad_h != qh || X.quad_fmt != (VkFormat)want
            || X.quad_stereo != stereo) {
        if (X.quad) {
            vkQueueWaitIdle(X.C->queue);
            for (i = 0; i < X.quad_n; i++) {
                if (X.quad_view[i]) vkDestroyImageView(X.C->dev, X.quad_view[i], NULL);
                X.quad_view[i] = VK_NULL_HANDLE;
            }
            xrDestroySwapchain(X.quad);
        }
        X.quad_has_frame = 0;
        X.quad = make_sc(want, scw, qh,
                         XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT
                         | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT);
        X.quad_n = X.quad ? enum_images(X.quad, X.quad_img) : 0;
        X.quad_w = qw;
        X.quad_h = qh;
        X.quad_fmt = (VkFormat)want;
        X.quad_stereo = stereo;
        if (draw)
            for (i = 0; i < X.quad_n; i++)
                X.quad_view[i] = view_of(X.quad_img[i], (VkFormat)want);
        update_size();
        fprintf(stderr, LOGX "screen swapchain %ux%u%s from a %dx%d picture, %s (%s)\n", qw, qh,
                stereo ? " x2 eyes" : "", w, h, draw ? "drawn" : "copied", X.quad ? "made" : "FAILED");
    }
    pthread_mutex_unlock(&X.quad_mu);
    if (!X.quad || !acquire(X.quad, &X.quad_idx) || X.quad_idx >= X.quad_n)
        return;
    X.quad_acquired = 1;
    X.quad_pose_pending = scene ? scene->pose : 0;
    if (draw && X.quad_view[X.quad_idx]) {
        VkDescriptorSet set = src_set(src, fmt);
        VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        VkRenderingInfo rb = {VK_STRUCTURE_TYPE_RENDERING_INFO};
        VkViewport v = {0, 0, (float)qw, (float)qh, 0, 1};
        VkRect2D sc = {{0, 0}, {scw, qh}};
        float r[4];
        VkDescriptorSet sset = stereo && scene ? ssrc_set(src, fmt, scene) : VK_NULL_HANDLE;
        barrier(cb, src, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        barrier(cb, X.quad_img[X.quad_idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        ca.imageView = X.quad_view[X.quad_idx];
        ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        ca.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        rb.renderArea = sc;
        rb.layerCount = 1;
        rb.colorAttachmentCount = 1;
        rb.pColorAttachments = &ca;
        vkCmdBeginRendering(cb, &rb);
        r[0] = (float)x / (float)sw;
        r[1] = (float)y / (float)sh;
        r[2] = (float)w / (float)sw;
        r[3] = (float)h / (float)sh;
        if (sset) {
            /* the two eyes, each from the scene and its depth */
            float pc[16];
            int e;
            int phys = X.immersive && scene->pose != 0;
            float tanh = phys ? X.pring[scene->pose % 64].tanh : 1.0f;
            VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            /* the backend's copies into the slot and its scene (earlier on
             * the queue) before the reads here */
            mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdEndRendering(cb);
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 1, &mb, 0, NULL, 0, NULL);
            vkCmdBeginRendering(cb, &rb);
            memcpy(pc, r, sizeof r);
            pc[4] = (float)scene->rect[0] / (float)scene->w;
            pc[5] = (float)scene->rect[1] / (float)scene->h;
            pc[6] = (float)scene->rect[2] / (float)scene->w;
            pc[7] = (float)scene->rect[3] / (float)scene->h;
            pc[9] = phys ? 1.0f : 0.35f;  /* screen: out of it at most 35% of the depth;
                                           * head-tracked: nothing nearer than 1 unit (1/z) */
            pc[10] = 3.0f / 255.0f;      /* a HUD pixel differs from the scene by more */
            pc[11] = 0.0f;
            pc[12] = 0.5f;               /* the race camera's near and far (xrcam.c's log) */
            pc[13] = 10000.0f;
            pc[14] = phys ? 1.0f : 0.0f;
            {
                static int dbg = -1;
                if (dbg < 0) {
                    const char *ev = getenv("RECOMP_XR_DEBUG");
                    dbg = !ev ? 0 : strstr(ev, "depth") ? 2 : strstr(ev, "mask") ? 1 : 0;
                }
                pc[15] = (float)dbg;
            }
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, X.spipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, X.spl, 0, 1, &sset, 0, NULL);
            for (e = 0; e < 2; e++) {
                static float dx = -1.0f;      /* RECOMP_XR_DEPTH_X: a test multiplier */
                if (dx < 0.0f) {
                    const char *ev = getenv("RECOMP_XR_DEPTH_X");
                    dx = ev && atof(ev) > 0.0 ? (float)atof(ev) : 1.0f;
                }
                if (phys)
                    /* an eye IPD/2 to the side (1 unit = 1 m): its picture moves the other way,
                     * by (IPD/2) / z of the half width 2 tan(h/2) */
                    pc[8] = (e ? -1.0f : 1.0f) * 0.032f * dx / (2.0f * tanh);
                else
                    pc[8] = (e ? 0.5f : -0.5f) * k_depth[X.depth_level] * dx;   /* half each way */
                v.x = (float)(e * qw);
                vkCmdPushConstants(cb, X.spl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 64, pc);
                vkCmdSetViewport(cb, 0, 1, &v);
                vkCmdSetScissor(cb, 0, 1, &sc);
                vkCmdDraw(cb, 3, 1, 0, 0);
            }
            X.frames3d++;
        } else if (set) {
            /* flat: the picture in each half (no scene this frame: menus, films) */
            int e;
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, X.pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, X.pl, 0, 1, &set, 0, NULL);
            vkCmdPushConstants(cb, X.pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, r);
            for (e = 0; e < (stereo ? 2 : 1); e++) {
                v.x = (float)(e * qw);
                vkCmdSetViewport(cb, 0, 1, &v);
                vkCmdSetScissor(cb, 0, 1, &sc);
                vkCmdDraw(cb, 3, 1, 0, 0);
            }
            X.frames2d++;
        }
        vkCmdEndRendering(cb);
        barrier(cb, src, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        if (sset && X.immersive && scene->pose && X.hpipe)
            hud_pass(cb, sset, scene);
        return;
    }
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
    if (X.hud_acquired) {
        X.hud_acquired = 0;
        release_locked(X.hud_sc);
        if (ok)
            X.hud_has = 1;
    }
    if (ok) {
        X.quad_has_frame = 1;
        X.quad_pose_id = X.quad_pose_pending;
    }
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
    const XrCompositionLayerBaseHeader *layers[6];
    uint32_t nl = 0, idx[2] = {0, 0}, e, n = 0;
    XrCompositionLayerPassthroughFB ptl = {XR_TYPE_COMPOSITION_LAYER_PASSTHROUGH_FB};
    XrView views[2] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};
    XrCompositionLayerProjectionView pv[2] = {{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW},
                                              {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}};
    XrCompositionLayerProjection proj = {XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    XrCompositionLayerQuad ql = {XR_TYPE_COMPOSITION_LAYER_QUAD}, ml = {XR_TYPE_COMPOSITION_LAYER_QUAD}, qr;
    XrCompositionLayerSettingsFB qs = {XR_TYPE_COMPOSITION_LAYER_SETTINGS_FB};
    XrFrameEndInfo fei = {XR_TYPE_FRAME_END_INFO};
    int mr = X.pt_layer && !X.dark, show_menu = X.menu_open && X.menu_sc && X.fs.shouldRender;
    XrResult r;
    /* head-tracked: the picture drawn with a pose, as a projection layer with
     * the eye poses it was drawn for (the menus and films stay on the screen) */
    int imm = X.immersive && X.quad && X.quad_has_frame && X.quad_pose_id && X.quad_stereo
              && X.fs.shouldRender;
    XrCompositionLayerProjectionView iv[2] = {{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW},
                                              {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}};
    XrCompositionLayerProjection ip = {XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    XrCompositionLayerQuad hl = {XR_TYPE_COMPOSITION_LAYER_QUAD};
    if (imm)
        mr = 0;

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
    if (X.fs.shouldRender && !imm) {
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
    if (imm) {
        uint32_t id = X.quad_pose_id, e2;
        float th, tv;
        pthread_mutex_lock(&X.pose_mu);
        th = X.pring[id % 64].tanh;
        tv = X.pring[id % 64].tanv > 0.0f ? X.pring[id % 64].tanv
           : th * (float)X.quad_h / (float)(X.quad_w ? X.quad_w : 1);   /* as the game drew it */
        for (e2 = 0; e2 < 2; e2++) {
            iv[e2].pose = X.pring[id % 64].v[e2].pose;
            iv[e2].fov.angleLeft = -atanf(th);
            iv[e2].fov.angleRight = atanf(th);
            iv[e2].fov.angleUp = atanf(tv);
            iv[e2].fov.angleDown = -atanf(tv);
            iv[e2].subImage.swapchain = X.quad;
            iv[e2].subImage.imageRect.offset.x = (int32_t)(e2 * X.quad_w);
            iv[e2].subImage.imageRect.extent.width = (int32_t)X.quad_w;
            iv[e2].subImage.imageRect.extent.height = (int32_t)X.quad_h;
        }
        pthread_mutex_unlock(&X.pose_mu);
        ip.space = X.local;
        ip.viewCount = 2;
        ip.views = iv;
        layers[nl++] = (const XrCompositionLayerBaseHeader *)&ip;
        X.imm_frames++;
        if (X.hud_sc && X.hud_has) {
            /* the HUD panel: 1.1 m ahead of the seat, 0.2 m below the eyes,
             * tipped back to face them (GT3 Revisited's hud_distance/drop) */
            const float tip = 0.18f;
            XrQuaternionf qt = {sinf(tip * 0.5f), 0.0f, 0.0f, cosf(tip * 0.5f)};
            V3 fwd, at;
            pthread_mutex_lock(&X.pose_mu);
            fwd = qrot(X.org_q, v3(0, 0, -1));
            at = vadd(X.org_pos, vadd(vmul(fwd, 1.1f), v3(0, -0.2f, 0)));
            hl.pose.orientation = qmul(X.org_q, qt);
            pthread_mutex_unlock(&X.pose_mu);
            hl.pose.position = at;
            hl.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
            hl.space = X.local;
            hl.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
            hl.subImage.swapchain = X.hud_sc;
            hl.subImage.imageRect.extent.width = HUD_W;
            hl.subImage.imageRect.extent.height = HUD_H;
            hl.size.width = 1.25f;
            hl.size.height = 1.25f * (float)HUD_H / (float)HUD_W;
            layers[nl++] = (const XrCompositionLayerBaseHeader *)&hl;
        }
    } else if (X.quad && X.quad_has_frame && X.fs.shouldRender) {
        if (X.ext_layer_settings) {
            qs.layerFlags = XR_COMPOSITION_LAYER_SETTINGS_QUALITY_SUPER_SAMPLING_BIT_FB
                          | XR_COMPOSITION_LAYER_SETTINGS_NORMAL_SHARPENING_BIT_FB;
            ql.next = &qs;
        }
        ql.space = X.local;
        ql.eyeVisibility = X.quad_stereo ? XR_EYE_VISIBILITY_LEFT : XR_EYE_VISIBILITY_BOTH;
        ql.subImage.swapchain = X.quad;
        ql.subImage.imageRect.extent.width = (int32_t)X.quad_w;
        ql.subImage.imageRect.extent.height = (int32_t)X.quad_h;
        ql.pose = X.quad_pose;
        ql.size = X.quad_size;
        layers[nl++] = (const XrCompositionLayerBaseHeader *)&ql;
        if (X.quad_stereo) {
            qr = ql;
            qr.eyeVisibility = XR_EYE_VISIBILITY_RIGHT;
            qr.subImage.imageRect.offset.x = (int32_t)X.quad_w;
            layers[nl++] = (const XrCompositionLayerBaseHeader *)&qr;
        }
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
        if (X.immersive) {
            /* the poses the game's next picture is drawn with: two display
             * periods ahead (it is drawn, then shown a frame or two later);
             * the compositor's reprojection takes care of the rest */
            XrViewLocateInfo vli = {XR_TYPE_VIEW_LOCATE_INFO};
            XrViewState vs = {XR_TYPE_VIEW_STATE};
            XrSpaceLocation hl = {XR_TYPE_SPACE_LOCATION};
            XrView vv[2] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};
            uint32_t nv = 0;
            XrTime t = X.fs.predictedDisplayTime + 2 * X.fs.predictedDisplayPeriod;
            vli.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
            vli.displayTime = t;
            vli.space = X.local;
            if (XR_SUCCEEDED(xrLocateViews(X.session, &vli, &vs, 2, &nv, vv)) && nv == 2
                    && XR_SUCCEEDED(xrLocateSpace(X.view, X.local, t, &hl))
                    && (hl.locationFlags & LOC_OK) == LOC_OK) {
                pthread_mutex_lock(&X.pose_mu);
                X.lv[0] = vv[0];
                X.lv[1] = vv[1];
                X.lhead = hl.pose;
                X.lvalid = 1;
                pthread_mutex_unlock(&X.pose_mu);
            }
        }
        if (X.motor[0] != X.last_hap[0] || X.motor[1] != X.last_hap[1] || X.motor[0] || X.motor[1])
            refresh_haptics(X.motor[0] != X.last_hap[0] || X.motor[1] != X.last_hap[1]);
        if (X.recenter)
            do_recenter(X.fs.predictedDisplayTime);
        end_frame();
        if (++frames % 1200 == 0) {
            uint64_t t = now_ms();
            fprintf(stderr, LOGX "frame thread: %.1f frames/s, screen %ux%u%s, %s, 3D %s (game frames: %llu 3D, %llu flat)%s\n",
                    1200.0 * 1000.0 / (double)(t - t0 ? t - t0 : 1), X.quad_w, X.quad_h,
                    X.quad_stereo ? " x2 eyes" : "", X.dark ? "dark room" : "mixed reality",
                    k_depth_name[X.depth_level], (unsigned long long)X.frames3d,
                    (unsigned long long)X.frames2d, X.immersive ? (X.imm_frames ? ", immersive: head-tracked layers" : ", immersive: no posed picture yet") : "");
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
    X.last_depth_on = X.depth_level ? X.depth_level : 1;
    nvk_xr_scene_wanted = X.depth_level > 0 || X.immersive;
    pthread_mutex_init(&X.pose_mu, NULL);
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
void xrt_quad_record(VkCommandBuffer cb, VkImage src, VkFormat fmt, int x, int y, int w, int h,
                     int sampled, uint32_t sw, uint32_t sh, const struct xrt_scene *scene)
{ (void)scene; (void)cb; (void)src; (void)fmt; (void)x; (void)y; (void)w; (void)h; (void)sampled; (void)sw; (void)sh; }
void xrt_quad_submitted(int ok) { (void)ok; }
void xrt_shutdown(void) {}

#endif
