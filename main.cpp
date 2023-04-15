#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/IconsFontAwesome5.h"
#include <stdio.h>
#include"myScene.h"
#include"myPipeline.h"
//#include"myPipeline.h"

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

float screenWidth = 1920, screenHeight = 960;
bool playMode = false;
double mouseX = 0, mouseY = 0;
double lastX = 0, lastY = 0; bool firstMouse = true;
double lastX_e_r = 0, lastY_e_r = 0; bool firstMouse_e_r = true;
double lastX_e_m = 0, lastY_e_m = 0; bool firstMouse_e_m = true;
double lastX_e_s = 0, lastY_e_s = 0; bool firstMouse_e_s = true;
GLfloat deltaTime = 0.0f;
GLfloat lastFrame = 0.0f;
myCamera *camera;
myScene scene;

int tabs = 0;
int object_parent = 0, object_leaf = -1;
int light_cur = 0;

/*重构窗口大小*/
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    screenWidth = width;
    screenHeight = height;
    glViewport(0, 0, width, height);
}
/*键盘输入响应函数*/
void processInput(GLFWwindow* window)
{
    if (playMode) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        float cameraSpeed = 0.002f / deltaTime; // adjust accordingly
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            camera->cameraPos += cameraSpeed * camera->cameraFront;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            camera->cameraPos -= cameraSpeed * camera->cameraFront;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            camera->cameraPos -= glm::normalize(glm::cross(camera->cameraFront, camera->cameraUp)) * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            camera->cameraPos += glm::normalize(glm::cross(camera->cameraFront, camera->cameraUp)) * cameraSpeed;
    }
}
/*鼠标事件回调函数*/
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    mouseX = xpos;
    mouseY = ypos;
    if (playMode) {
        if (firstMouse)
        {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
            return;
        }
        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos;
        lastX = xpos;
        lastY = ypos;
        float sensitivity = 0.1;
        xoffset *= sensitivity;
        yoffset *= sensitivity;
        camera->cameraEuler.yaw += xoffset;
        camera->cameraEuler.pitch += yoffset;
        if (camera->cameraEuler.pitch > 89.0f)
            camera->cameraEuler.pitch = 89.0f;
        if (camera->cameraEuler.pitch < -89.0f)
            camera->cameraEuler.pitch = -89.0f;
        glm::vec3 front;
        front.x = cos(glm::radians(camera->cameraEuler.yaw)) * cos(glm::radians(camera->cameraEuler.pitch));
        front.y = sin(glm::radians(camera->cameraEuler.pitch));
        front.z = sin(glm::radians(camera->cameraEuler.yaw)) * cos(glm::radians(camera->cameraEuler.pitch));
        camera->cameraFront = glm::normalize(front);
    }
    else {
        if (xpos < 375 || xpos>1474 || ypos < 66 || ypos>729) {
            firstMouse_e_r = true;
            firstMouse_e_m = true;
            firstMouse_e_s = true;
            return;
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        {
            if (firstMouse_e_r)
            {
                lastX_e_r = xpos;
                lastY_e_r = ypos;
                firstMouse_e_r = false;
                return;
            }
            float xoffset = xpos - lastX_e_r;
            float yoffset = lastY_e_r - ypos;
            lastX_e_r = xpos;
            lastY_e_r = ypos;
            float sensitivity = 0.2;
            xoffset *= sensitivity;
            yoffset *= sensitivity;
            camera->cameraEuler.yaw += xoffset;
            camera->cameraEuler.pitch += yoffset;
            if (camera->cameraEuler.pitch > 89.0f)
                camera->cameraEuler.pitch = 89.0f;
            if (camera->cameraEuler.pitch < -89.0f)
                camera->cameraEuler.pitch = -89.0f;
            glm::vec3 front;
            front.x = cos(glm::radians(camera->cameraEuler.yaw)) * cos(glm::radians(camera->cameraEuler.pitch));
            front.y = sin(glm::radians(camera->cameraEuler.pitch));
            front.z = sin(glm::radians(camera->cameraEuler.yaw)) * cos(glm::radians(camera->cameraEuler.pitch));
            camera->cameraFront = glm::normalize(front);
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS)
        {
            if (firstMouse_e_m)
            {
                lastX_e_m = xpos;
                lastY_e_m = ypos;
                firstMouse_e_m = false;
                return;
            }
            float xoffset = xpos - lastX_e_m;
            float yoffset = lastY_e_m - ypos;
            lastX_e_m = xpos;
            lastY_e_m = ypos;
            float sensitivity = 0.08;
            xoffset *= sensitivity;
            yoffset *= sensitivity;
            camera->cameraPos += glm::normalize(glm::cross(camera->cameraFront, camera->cameraUp)) * xoffset;
            camera->cameraPos += glm::normalize(camera->cameraUp) * yoffset;
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
        {
            if (firstMouse_e_s)
            {
                lastX_e_s = xpos;
                lastY_e_s = ypos;
                firstMouse_e_s = false;
                return;
            }
            float xoffset = xpos - lastX_e_s;
            float yoffset = lastY_e_s - ypos;
            lastX_e_s = xpos;
            lastY_e_s = ypos;
            float sensitivity = 0.13;
            xoffset *= sensitivity;
            yoffset *= sensitivity;
            camera->cameraPos += glm::normalize(camera->cameraFront) * yoffset;
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
            firstMouse_e_r = true;
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_RELEASE) {
            firstMouse_e_m = true;
        }
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_RELEASE) {
            firstMouse_e_s = true;
        }
    }
}
/*滚轮事件回调函数*/
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (playMode) {
        float sensitivity = 0.1;
        if (camera->fov >= 1.0f && camera->fov <= 45.0f)
            camera->fov -= yoffset * sensitivity;
        if (camera->fov <= 1.0f)
            camera->fov = 1.0f;
        if (camera->fov >= 45.0f)
            camera->fov = 45.0f;
    }
    else {
        if (mouseX < 375 || mouseX > 1474 || mouseY < 66 || mouseY > 729) {
            return;
        }
        float cameraSpeed = 0.02f / deltaTime;
        camera->cameraPos += cameraSpeed * float(yoffset) * camera->cameraFront;
    }
}

bool show_settings_window = false;

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}
static void ShowExampleMenuFile()
{
    //IMGUI_DEMO_MARKER("Examples/Menu");
    //ImGui::MenuItem("(demo menu)", NULL, false, false);
    if (ImGui::MenuItem("New")) {}
    if (ImGui::MenuItem("Open", "Ctrl+O")) {}
    if (ImGui::BeginMenu("Open Recent"))
    {
        ImGui::MenuItem("fish_hat.c");
        ImGui::MenuItem("fish_hat.inl");
        ImGui::MenuItem("fish_hat.h");
        if (ImGui::BeginMenu("More.."))
        {
            ImGui::MenuItem("Hello");
            ImGui::MenuItem("Sailor");
            if (ImGui::BeginMenu("Recurse.."))
            {
                ShowExampleMenuFile();
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenu();
    }
    if (ImGui::MenuItem("Save", "Ctrl+S")) {}
    if (ImGui::MenuItem("Save As..")) {}

    ImGui::Separator();
    //IMGUI_DEMO_MARKER("Examples/Menu/Options");

    if (ImGui::BeginMenu("Options"))
    {
        static bool enabled = true;
        ImGui::MenuItem("Enabled", "", &enabled);
        ImGui::BeginChild("child", ImVec2(0, 60), true);
        for (int i = 0; i < 10; i++)
            ImGui::Text("Scrolling Text %d", i);
        ImGui::EndChild();
        static float f = 0.5f;
        static int n = 0;
        ImGui::SliderFloat("Value", &f, 0.0f, 1.0f);
        ImGui::InputFloat("Input", &f, 0.1f);
        ImGui::Combo("Combo", &n, "Yes\0No\0Maybe\0\0");
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Colors"))
    {
        float sz = ImGui::GetTextLineHeight();
        for (int i = 0; i < ImGuiCol_COUNT; i++)
        {
            const char* name = ImGui::GetStyleColorName((ImGuiCol)i);
            ImVec2 p = ImGui::GetCursorScreenPos();
            ImGui::GetWindowDrawList()->AddRectFilled(p, ImVec2(p.x + sz, p.y + sz), ImGui::GetColorU32((ImGuiCol)i));
            ImGui::Dummy(ImVec2(sz, sz));
            ImGui::SameLine();
            ImGui::MenuItem(name);
        }
        ImGui::EndMenu();
    }

    // Here we demonstrate appending again to the "Options" menu (which we already created above)
    // Of course in this demo it is a little bit silly that this function calls BeginMenu("Options") twice.
    // In a real code-base using it would make senses to use this feature from very different code locations.
    if (ImGui::BeginMenu("Options")) // <-- Append!
    {
        //IMGUI_DEMO_MARKER("Examples/Menu/Append to an existing menu");
        static bool b = true;
        ImGui::Checkbox("SomeOption", &b);
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Disabled", false)) // Disabled
    {
        IM_ASSERT(0);
    }
    if (ImGui::MenuItem("Checked", NULL, true)) {}
    ImGui::Separator();
    if (ImGui::MenuItem("Quit", "Alt+F4")) {}
}
static void ShowExampleAppMainMenuBar()
{
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            ShowExampleMenuFile();
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit"))
        {
            if (ImGui::MenuItem("Undo", "CTRL+Z")) {}
            if (ImGui::MenuItem("Redo", "CTRL+Y", false, false)) {}  // Disabled item
            ImGui::Separator();
            if (ImGui::MenuItem("Cut", "CTRL+X")) {}
            if (ImGui::MenuItem("Copy", "CTRL+C")) {}
            if (ImGui::MenuItem("Paste", "CTRL+V")) {}
            ImGui::Separator();
            if (ImGui::MenuItem("Settings", "CTRL+Q")) {
                show_settings_window = true;
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Asset"))
        {
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Window"))
        {
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("About"))
        {
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

static void ShowSceneObj() {
    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoResize;
    ImGui::Begin("Scene", NULL, window_flags);
    if (ImGui::BeginTabBar("tabs", ImGuiTabBarFlags_None))
    {
        if (ImGui::BeginTabItem("Objects"))
        {
            tabs = 0;
            for (int i = 0; i < scene.models.size(); i++) {
                char tmp[10] = "";
                char goal[30];
                printf_s(tmp, "%d_%d", i, -1);
                const char* pre = ICON_FA_CUBES "     \0";
                goal[0] = '\0';
                strcpy(goal, pre);
                strcat(goal, scene.models[i].name.data());

                ImGuiTreeNodeFlags flags = (i == object_parent && object_leaf == -1 ? ImGuiTreeNodeFlags_Selected : 0)
                    | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;
                bool opened = ImGui::TreeNodeEx(tmp, flags, goal);
                if (ImGui::IsItemClicked(0)) {
                    object_parent = i;
                    object_leaf = -1;
                }
                if (opened) {
                    for (int j = 0; j < scene.models[i].meshes.size(); j++) {
                        char tmp2[10] = "";
                        char goal2[150];
                        printf_s(tmp2, "%d_%d", i, j);
                        const char* pre2 = ICON_FA_CUBE "     \0";
                        goal2[0] = '\0';
                        strcpy(goal2, pre2);
                        strcat(goal2, scene.models[i].meshes[j].name.data());
                        flags = (i == object_parent && j == object_leaf ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_Leaf;
                        if (ImGui::TreeNodeEx(tmp2, flags, goal2)) {
                            if (ImGui::IsItemClicked(0)) {
                                object_parent = i;
                                object_leaf = j;
                            }
                            ImGui::TreePop();
                        }
                    }
                    ImGui::TreePop();
                }
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Lights"))
        {
            tabs = 1;
            char* item[20];
            char goal[20][50];
            for (int i = 0; i < scene.lights.size(); i++) {
                const char* pre = ICON_FA_LIGHTBULB "     \0";
                goal[i][0] = '\0';
                strcpy(goal[i], pre);
                strcat(goal[i], scene.lights[i]->getName().data());
                item[i] = goal[i];
            }
            ImGui::ListBox("", &light_cur, item, scene.lights.size(), 19);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Cameras"))
        {
            tabs = 2;
            char* item[20];
            char goal[20][50];
            for (int i = 0; i < scene.cameras.size(); i++) {
                const char* pre = ICON_FA_LIGHTBULB "     \0";
                goal[i][0] = '\0';
                strcpy(goal[i], pre);
                strcat(goal[i], scene.cameras[i].name.data());
                item[i] = goal[i];
            }
            static int light_current = 0;
            ImGui::ListBox("", &light_current, item, scene.cameras.size(), 19);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

static void ShowResources() {
    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoResize;
    ImGui::Begin("Resources", NULL, window_flags);
    if (ImGui::BeginTabBar("tabs", ImGuiTabBarFlags_None))
    {
        if (ImGui::BeginTabItem("Textures"))
        {
            for (int i = 0; i < scene.textures.size(); i++) {
                ImGui::Image((GLuint*)scene.textures[i].id, ImVec2(80, 80), ImVec2(0, 0), ImVec2(1, 1), ImVec4(1, 1, 1, 1), ImVec4(1, 1, 1, 1)); 
                ImGui::SameLine(); 
                ImGui::Text(scene.textures[i].path.data());
                ImGui::Separator();
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Shaders"))
        {
            char* item[20];
            char goal[20][50];
            for (int i = 0; i < scene.shaders.size(); i++) {
                const char* pre = ICON_FA_FILE_CODE "     \0";
                goal[i][0] = '\0';
                strcpy(goal[i], pre);
                strcat(goal[i], scene.shaders[i].name.data());
                item[i] = goal[i];
            }
            static int item_current = 0;
            ImGui::ListBox("", &item_current, item, scene.shaders.size(), 18);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Scripts"))
        {
            const char* items[] = { ICON_FA_FILE_CODE "     translate.lua",
                ICON_FA_FILE_CODE "     rotate.lua", };
            static int item_current = 0;
            ImGui::ListBox("", &item_current, items, IM_ARRAYSIZE(items), 18);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

static void ShowRenderSettings() {
    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoResize;
    window_flags |= ImGuiWindowFlags_NoTitleBar;
    ImGui::Begin("RenderSettings", NULL, window_flags);
    bool f = false;
    if (ImGui::Checkbox("DepthMode", &f)) {}
    ImGui::SameLine();
    if (ImGui::Checkbox("LineMode", &f)) {}
    ImGui::SameLine();
    if (ImGui::Checkbox("ShowMessage", &f)) {}
    ImGui::SameLine();
    if (ImGui::Checkbox("HideWindows", &f)) {}
    ImGui::SameLine();
    ImGui::End();
}

static void ShowNodeProperties() {
    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoResize;
    ImGui::Begin("NodeProperties", NULL, window_flags);
    if (ImGui::BeginTabBar("tabs", ImGuiTabBarFlags_None))
    {
        if (ImGui::BeginTabItem("Instance"))
        {
            if (tabs == 0) {
                myModel* cur = &scene.models[object_parent];
                bool h = false;
                ImVec4 clear_color[5];
                ImGui::Checkbox("Hide", &h);
                if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
                    float* posx, * posy, * posz, * rotx, * roty, * rotz, * scalex, * scaley, * scalez;
                    if (object_leaf == -1) {
                        posx = &cur->pos.x; posy = &cur->pos.y; posz = &cur->pos.z;
                        rotx = &cur->rot.x; roty = &cur->rot.y; rotz = &cur->rot.z;
                        scalex = &cur->scale.x; scaley = &cur->scale.y; scalez = &cur->scale.z;
                    }
                    else {
                        posx = &cur->meshes[object_leaf].pos.x; posy = &cur->meshes[object_leaf].pos.y; posz = &cur->meshes[object_leaf].pos.z;
                        rotx = &cur->meshes[object_leaf].rot.x; roty = &cur->meshes[object_leaf].rot.y; rotz = &cur->meshes[object_leaf].rot.z;
                        scalex = &cur->meshes[object_leaf].scale.x; scaley = &cur->meshes[object_leaf].scale.y; scalez = &cur->meshes[object_leaf].scale.z;
                    }
                    ImGui::Separator();
                    ImGui::Text("Position");
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
                    ImGui::Button("X"); ImGui::SameLine(); ImGui::DragFloat(" ##posX", posx); ImGui::SameLine();
                    ImGui::PopStyleColor();
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.8f, 0.1f, 1.0f));
                    ImGui::Button("Y"); ImGui::SameLine(); ImGui::DragFloat(" ##posY", posy); ImGui::SameLine();
                    ImGui::PopStyleColor();
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.8f, 1.0f));
                    ImGui::Button("Z"); ImGui::SameLine(); ImGui::DragFloat(" ##posZ", posz);
                    ImGui::PopStyleColor();
                    ImGui::Separator();

                    ImGui::Text("Rotation");
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
                    ImGui::Button("X"); ImGui::SameLine(); ImGui::DragFloat(" ##rotX", rotx); ImGui::SameLine();
                    ImGui::PopStyleColor();
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.8f, 0.1f, 1.0f));
                    ImGui::Button("Y"); ImGui::SameLine(); ImGui::DragFloat(" ##rotY", roty); ImGui::SameLine();
                    ImGui::PopStyleColor();
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.8f, 1.0f));
                    ImGui::Button("Z"); ImGui::SameLine(); ImGui::DragFloat(" ##rotZ", rotz);
                    ImGui::PopStyleColor();
                    ImGui::Separator();

                    ImGui::Text("Scale");
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
                    ImGui::Button("X"); ImGui::SameLine(); ImGui::DragFloat(" ##scaX", scalex); ImGui::SameLine();
                    ImGui::PopStyleColor();
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.8f, 0.1f, 1.0f));
                    ImGui::Button("Y"); ImGui::SameLine(); ImGui::DragFloat(" ##scaY", scaley); ImGui::SameLine();
                    ImGui::PopStyleColor();
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.8f, 1.0f));
                    ImGui::Button("Z"); ImGui::SameLine(); ImGui::DragFloat(" ##scaZ", scalez);
                    ImGui::PopStyleColor();
                    ImGui::Separator();
                    ImGui::Button("To Center", ImVec2(100, 25));
                }
                if (object_leaf != -1) {
                    if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
                        const char* items[10];
                        for (int i = 0; i < scene.shaders.size(); i++)
                            items[i] = scene.shaders[i].name.data();
                        ImGui::Combo("Shader", &scene.models[object_parent].meshes[object_leaf].use_shader, items, scene.shaders.size());
                        ImGui::Separator();
                        ImTextureID dif = NULL, met = NULL, rou = NULL, nor = NULL, spe = NULL, ao = NULL;
                        for (int i = 0; i < cur->meshes[object_leaf].texture_struct.size(); i++) {
                            GLuint* tex = (GLuint*)(cur->meshes[object_leaf].texture_struct[i].id);
                            if (cur->meshes[object_leaf].texture_struct[i].type == "diffuse_texture") dif = tex;
                            if (cur->meshes[object_leaf].texture_struct[i].type == "roughness_texture") rou = tex;
                            if (cur->meshes[object_leaf].texture_struct[i].type == "metallic_texture") met = tex;
                            if (cur->meshes[object_leaf].texture_struct[i].type == "normal_texture") nor = tex;
                            if (cur->meshes[object_leaf].texture_struct[i].type == "specular_texture") spe = tex;
                            if (cur->meshes[object_leaf].texture_struct[i].type == "ambient_texture") ao = tex;
                        }
                        ImGui::Text("Diffuse:");
                        ImGui::ColorEdit4("dif", (float*)&cur->meshes[object_leaf].diffuse);
                        ImGui::Image(dif, ImVec2(120, 120), ImVec2(0, 0), ImVec2(1, 1), ImVec4(1, 1, 1, 1), ImVec4(1, 1, 1, 1));
                        ImGui::Separator();
                        ImGui::Text("Metallic:");
                        ImGui::SliderFloat("met", &cur->meshes[object_leaf].metallic, 0, 1);
                        ImGui::Image(met, ImVec2(120, 120), ImVec2(0, 0), ImVec2(1, 1), ImVec4(1, 1, 1, 1), ImVec4(1, 1, 1, 1));
                        ImGui::Separator();
                        ImGui::Text("Roughness:");
                        ImGui::SliderFloat("rou", &cur->meshes[object_leaf].roughness, 0, 1);
                        ImGui::Image(rou, ImVec2(120, 120), ImVec2(0, 0), ImVec2(1, 1), ImVec4(1, 1, 1, 1), ImVec4(1, 1, 1, 1));
                        ImGui::Separator();
                        ImGui::Text("Normal:");
                        ImGui::Image(nor, ImVec2(120, 120), ImVec2(0, 0), ImVec2(1, 1), ImVec4(1, 1, 1, 1), ImVec4(1, 1, 1, 1));
                        ImGui::Separator();
                        ImGui::Text("Specular:");
                        ImGui::ColorEdit3("spe", (float*)&cur->meshes[object_leaf].specular);
                        ImGui::Image(spe, ImVec2(120, 120), ImVec2(0, 0), ImVec2(1, 1), ImVec4(1, 1, 1, 1), ImVec4(1, 1, 1, 1));
                        ImGui::Separator();
                        ImGui::Text("Ambient Occlusion:");
                        ImGui::SliderFloat("ao", &cur->meshes[object_leaf].ambient, 0, 1);
                        ImGui::Image(ao, ImVec2(120, 120), ImVec2(0, 0), ImVec2(1, 1), ImVec4(1, 1, 1, 1), ImVec4(1, 1, 1, 1));
                        ImGui::Separator();
                    }
                }
                if (ImGui::CollapsingHeader("Scripts", ImGuiTreeNodeFlags_DefaultOpen)) {

                }
            }
            else if (tabs == 1)
            {
                if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
                    myLight* light = scene.lights[light_cur];
                    int kind = light->getKind();
                    if (kind == 0) {
                        ImGui::Text("Position");
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
                        ImGui::Button("X"); ImGui::SameLine(); ImGui::DragFloat(" ##lposX", &light->position.x); ImGui::SameLine();
                        ImGui::PopStyleColor();
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.8f, 0.1f, 1.0f));
                        ImGui::Button("Y"); ImGui::SameLine(); ImGui::DragFloat(" ##lposY", &light->position.y); ImGui::SameLine();
                        ImGui::PopStyleColor();
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.8f, 1.0f));
                        ImGui::Button("Z"); ImGui::SameLine(); ImGui::DragFloat(" ##lposZ", &light->position.z);
                        ImGui::PopStyleColor();
                        ImGui::Separator();

                        ImGui::Text("Goal");
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
                        ImGui::Button("X"); ImGui::SameLine(); ImGui::DragFloat(" ##lrotX", &light->direction.x); ImGui::SameLine();
                        ImGui::PopStyleColor();
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.8f, 0.1f, 1.0f));
                        ImGui::Button("Y"); ImGui::SameLine(); ImGui::DragFloat(" ##lrotY", &light->direction.y); ImGui::SameLine();
                        ImGui::PopStyleColor();
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.8f, 1.0f));
                        ImGui::Button("Z"); ImGui::SameLine(); ImGui::DragFloat(" ##lrotZ", &light->direction.z);
                        ImGui::PopStyleColor();
                        ImGui::Separator();

                        ImGui::Text("Color:");
                        ImGui::ColorEdit3("dif", (float*)&light->color);
                        ImGui::DragFloat(" ##lintensity", &light->intensity);
                        ImGui::Separator();

                        ImGui::Text("Plane");
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
                        ImGui::Button("X"); ImGui::SameLine(); ImGui::DragFloat(" ##lplaneX", &light->plane.x); ImGui::SameLine();
                        ImGui::PopStyleColor();
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.8f, 0.1f, 1.0f));
                        ImGui::Button("Y"); ImGui::SameLine(); ImGui::DragFloat(" ##lplaneY", &light->plane.y); ImGui::SameLine();
                        ImGui::PopStyleColor();
                        ImGui::Separator();

                        ImGui::Text("Ortho");
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
                        ImGui::Button("left"); ImGui::SameLine(); ImGui::DragFloat(" ##lorthoX", &light->ortho.x); ImGui::SameLine();
                        ImGui::PopStyleColor();
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.8f, 0.1f, 1.0f));
                        ImGui::Button("right"); ImGui::SameLine(); ImGui::DragFloat(" ##lorthoY", &light->ortho.y);
                        ImGui::PopStyleColor();
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.8f, 1.0f));
                        ImGui::Button("bottom"); ImGui::SameLine(); ImGui::DragFloat(" ##lorthoZ", &light->ortho.z); ImGui::SameLine();
                        ImGui::PopStyleColor();
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.8f, 0.1f, 1.0f));
                        ImGui::Button("top"); ImGui::SameLine(); ImGui::DragFloat(" ##lorthoW", &light->ortho.w);
                        ImGui::PopStyleColor();
                        ImGui::Separator();
                        /*ImGui::Text("Shadowmap Resolution");
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
                        ImGui::Button("X"); ImGui::SameLine(); ImGui::DragFloat(" ##lshadowX", &light->shadowmap_resolution.x); ImGui::SameLine();
                        ImGui::PopStyleColor();
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.8f, 0.1f, 1.0f));
                        ImGui::Button("Y"); ImGui::SameLine(); ImGui::DragFloat(" ##lshadowY", &light->shadowmap_resolution.y); ImGui::SameLine();
                        ImGui::PopStyleColor();*/
                    }
                    else {
                        ImGui::Text("Position");
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
                        ImGui::Button("X"); ImGui::SameLine(); ImGui::DragFloat(" ##lposX", &light->position.x); ImGui::SameLine();
                        ImGui::PopStyleColor();
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.8f, 0.1f, 1.0f));
                        ImGui::Button("Y"); ImGui::SameLine(); ImGui::DragFloat(" ##lposY", &light->position.y); ImGui::SameLine();
                        ImGui::PopStyleColor();
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.8f, 1.0f));
                        ImGui::Button("Z"); ImGui::SameLine(); ImGui::DragFloat(" ##lposZ", &light->position.z);
                        ImGui::PopStyleColor();
                        ImGui::Separator();

                        ImGui::Text("Color:");
                        ImGui::ColorEdit3("dif", (float*)&light->color);
                        ImGui::DragFloat(" ##lintensity", &light->intensity);
                        ImGui::Separator();

                        ImGui::Text("Attenuation");
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
                        ImGui::Button("C"); ImGui::SameLine(); ImGui::DragFloat(" ##lorthoX", &light->constant, 1, 10); ImGui::SameLine();
                        ImGui::PopStyleColor();
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.8f, 0.1f, 1.0f));
                        ImGui::Button("L"); ImGui::SameLine(); ImGui::DragFloat(" ##lorthoY", &light->linear, 0, 1); ImGui::SameLine();
                        ImGui::PopStyleColor();
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
                        ImGui::Button("Q"); ImGui::SameLine(); ImGui::DragFloat(" ##lorthoZ", &light->quadratic, 0, 1); ImGui::SameLine();
                        ImGui::PopStyleColor();
                        ImGui::Separator();

                        ImGui::Text("Plane");
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
                        ImGui::Button("X"); ImGui::SameLine(); ImGui::DragFloat(" ##lplaneX", &light->plane.x); ImGui::SameLine();
                        ImGui::PopStyleColor();
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.8f, 0.1f, 1.0f));
                        ImGui::Button("Y"); ImGui::SameLine(); ImGui::DragFloat(" ##lplaneY", &light->plane.y); ImGui::SameLine();
                        ImGui::PopStyleColor();
                        ImGui::Separator();
                        /*ImGui::Text("Shadowmap Resolution");
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
                        ImGui::Button("X"); ImGui::SameLine(); ImGui::DragFloat(" ##lshadowX", &light->shadowmap_resolution.x); ImGui::SameLine();
                        ImGui::PopStyleColor();
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.8f, 0.1f, 1.0f));
                        ImGui::Button("Y"); ImGui::SameLine(); ImGui::DragFloat(" ##lshadowY", &light->shadowmap_resolution.y); ImGui::SameLine();
                        ImGui::PopStyleColor();*/
                    }
                }
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Environment"))
        {
            bool f;
            float v = 1;
            ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
            ImGui::Text("Environment Map:");
            ImGui::ColorEdit3("Map", (float*)&clear_color);
            ImGui::Image(NULL, ImVec2(120, 120), ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0), ImVec4(1, 1, 1, 1));
            ImGui::Separator();
            ImGui::Text("Post Processing:");
            ImGui::Checkbox("Gaussian Filter", &f);
            ImGui::Checkbox("Laplace Filter", &f);
            ImGui::SliderFloat("Saturation", &v, 0, 1);
            ImGui::SliderFloat("Lightness", &v, 0, 1);
            ImGui::Separator();
            ImGui::EndTabItem();
            /*const char* items[] = {"Directional", "Point"};
            static int light_type_current = 0;
            ImGui::Combo("Type", &light_type_current, items, 2);
            ImGui::EndTabItem();*/
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void ShowConsoles() {
    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoResize;
    ImGui::Begin("Consoles", NULL, window_flags);
    const float footer_height_to_reserve = ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing() + 10;
    if (ImGui::BeginChild("ScrollingRegion", ImVec2(0, -footer_height_to_reserve), false, ImGuiWindowFlags_HorizontalScrollbar))
    {

        ImGui::EndChild();
    }
    ImGui::Separator();
    char InputBuf[5000] = { '\0' };
    if (ImGui::InputText("Input", InputBuf, IM_ARRAYSIZE(InputBuf), 0, 0, (void*)0))
    {
        char* s = InputBuf;
    }
    ImGui::End();
}


int main()
{
    glfwSetErrorCallback(glfw_error_callback);
    /*opengl窗口初始化*/
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);                                //主版本：3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);                                //次版本：3
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);                //设置为核心模式
    glfwWindowHint(GLFW_MAXIMIZED, true);
    GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "PL_RealtimeRenderer", NULL, NULL);   //开启窗口
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);                                        //将窗口上下文绑定为当前线程的上下文
    glfwSwapInterval(1);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);     //绑定窗口大小改变时调用的函数

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))               //初始化glad
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    glfwSetCursorPosCallback(window, mouse_callback);
    //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetScrollCallback(window, scroll_callback);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    glfwWindowHint(GLFW_SAMPLES, 4);
    glEnable(GL_MULTISAMPLE);

    scene = myScene(0);
    camera = &(scene.cameras[0]);
    myPipeline pipeline = myPipeline(&scene);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();

    ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\Arial.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesChineseFull());

    ImFontConfig config;
    config.MergeMode = true;
    config.GlyphMinAdvanceX = 13.0f; // Use if you want to make the icon monospaced
    static const ImWchar icon_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
    io.Fonts->AddFontFromFileTTF("imgui/fonts/fontawesome-webfont.ttf", 17.0f, &config, icon_ranges);

    ImFont* font2 = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\Arial.ttf", 20.0f, NULL, io.Fonts->GetGlyphRangesChineseFull());
    // Our state
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        GLfloat currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        processInput(window);
        // Set Style
        ImGuiStyle& style = ImGui::GetStyle();
        style.Colors[2].w = 1;
        style.Colors[10].w = 1;
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ShowSceneObj();
        ShowResources();
        ShowRenderSettings();
        ShowNodeProperties();
        ShowConsoles();
        if (show_settings_window) {
            ImGui::Begin("Settings", &show_settings_window, 0);
            ImGui::ShowStyleEditor();
            ImGui::End();
        }
        ImGui::PushFont(font2);
        ShowExampleAppMainMenuBar();
        ImGui::PopFont();

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);

        pipeline.render();
        // Rendering
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwMakeContextCurrent(window);
        glfwSwapBuffers(window);               //双缓冲，交换前后buffer
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();                           //结束线程，释放资源
    return 0;
}