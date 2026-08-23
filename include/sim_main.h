#pragma once

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#define WINDOW_WIDTH (1400)
#define WINDOW_HEIGHT (1000)

#define MOUSE_SCALE (0.002)
#define ZOOM_SENSITIVITY (0.05)

// 物理は 1 kHz。Isaac の sim dt と同じ。
#define PHYSICS_TIMESTEP (0.001)
// policy は 62.5 Hz。Isaac の decimation 16 x 1 kHz と同じ。
#define CONTROL_TIMESTEP (0.016)
// モータドライバは物理と同じ 1 kHz で回す。
// Isaac の実装 PD も target を 16 物理 step 保持して 1 kHz で効くので、これで一致する。
#define DRIVER_TIMESTEP (PHYSICS_TIMESTEP)

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
