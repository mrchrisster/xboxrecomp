/**
 * xr_theater -- the game on a floating screen in a Meta Quest headset
 * (OpenXR, Android). A port of FuturamaRecomp's OpenXrTheater for the
 * toolkit's Vulkan presenter, drawn on the GPU: the presenter's finished
 * frame (the game and the settings overlay) is copied into an OpenXR quad
 * swapchain image in the present's own command buffer, nothing is read
 * back.
 *
 * Start-up order:
 *   xrt_preinit()     before nvk_core_init (fb_present_vulkan.c's fbv_create):
 *                     the loader, the XrInstance and the headset; on success the
 *                     Vulkan instance and device are made by the OpenXR runtime
 *                     (XR_KHR_vulkan_enable2, nvk_core_set_xr_hooks), so the
 *                     game and the compositor share one device and queue.
 *                     Off a headset (no runtime) it returns 0 and nothing
 *                     changes.
 *   xrt_start(C)      after the device: the session, spaces, Touch actions,
 *                     passthrough, the menu panel and the frame thread, which
 *                     runs xrWaitFrame/xrBeginFrame/xrEndFrame at the headset's
 *                     rate and composes passthrough (or a dark room), the screen
 *                     quad and the VR menu.
 * Per present (the presenter's thread):
 *   xrt_quad_record(cb, img, fmt, rect)  in the present's command buffer,
 *                     with img in TRANSFER_DST_OPTIMAL (left so): copies the
 *                     rect into the next quad image.
 *   xrt_quad_submitted(ok)  right after the present's vkQueueSubmit, under
 *                     nvk_queue_lock: releases the quad image.
 *
 * Touch controllers are pad 0 (input_core.c's XR source), the game's rumble
 * goes to the controllers' haptics. Controls (as Futurama's):
 *   both grips          window mode: the screen follows the hands, hands apart
 *                       pushes it away, L stick Y distance, R stick Y size;
 *                       + R3 dark room / mixed reality, + Y reset the screen
 *   L3 + R3             the VR menu (resume, background, recenter, reset, exit)
 *   Menu (left)         Start; in window mode Back
 * The placement is kept in <external files>/xr_view.txt.
 * RECOMP_XR=0 in env.txt turns it all off (the flat Android window).
 */
#ifndef XBOXRECOMP_XR_THEATER_H
#define XBOXRECOMP_XR_THEATER_H

#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

struct nvk_core;

int  xrt_preinit(void);
int  xrt_on(void);
int  xrt_start(const struct nvk_core *C);
/* A frame's 3D inputs (fb_present_vulkan.c): the slot (the frame with its
 * HUD, before the post passes), the scene before the HUD and its depth (R32
 * float), all w x h, with the picture at rect (x, y, w, h). */
struct xrt_scene {
    VkImage frame;
    VkImageView frame_view, scene_view, depth_view;
    uint32_t w, h;
    int rect[4];
};

/* sampled: img has SAMPLED usage (drawn, scaled to at most 1920 wide, by a
 * render pass); else copied as it is. sw, sh: img's size. */
void xrt_quad_record(VkCommandBuffer cb, VkImage src, VkFormat fmt,
                     int x, int y, int w, int h, int sampled, uint32_t sw, uint32_t sh,
                     const struct xrt_scene *scene);
void xrt_quad_submitted(int ok);
void xrt_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
