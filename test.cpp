#include <iostream>
//#include <glad/glad.h>
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/IconsFontAwesome5.h"
#include <stdio.h>
#include"myScene.h"
#include"myPipeline.h"
#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif
#include <GLFW/glfw3.h>

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

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
            const char* items[] = { ICON_FA_CUBE "     Apple",
                ICON_FA_CUBE "     Banana",
                ICON_FA_CUBE "     Cherry",
                ICON_FA_CUBE "     Kiwi",
                ICON_FA_CUBE "     Mango",
                ICON_FA_CUBE "     Orange",
                ICON_FA_CUBE "     Pineapple",
                ICON_FA_CUBE "     Strawberry",
                ICON_FA_CUBE "     Watermelon" };
            static int item_current = 0;
            ImGui::ListBox("", &item_current, items, IM_ARRAYSIZE(items), 19);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Lights"))
        {
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Cameras"))
        {
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
            for (int i = 0; i < 4; i++) {
                ImGui::Image(NULL, ImVec2(80, 80), ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0), ImVec4(1, 1, 1, 1)); ImGui::SameLine(); ImGui::Text("123");
                ImGui::Separator();
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Shaders"))
        {
            const char* items[] = { ICON_FA_FILE_CODE "     light_shader",
                ICON_FA_FILE_CODE "     phong_shader", };
            static int item_current = 0;
            ImGui::ListBox("", &item_current, items, IM_ARRAYSIZE(items), 18);
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
            float f = 0;
            bool h = false;
            ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
            ImVec4 clear_color2 = ImVec4(0.0f, 0.0f, 1.00f, 1.00f);
            ImGui::Checkbox("Hide", &h);
            if (ImGui::CollapsingHeader("Transform")) {
                ImGui::Separator();
                ImGui::Text("Position");
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
                ImGui::Button("X"); ImGui::SameLine(); ImGui::DragFloat(" ##posX", &f); ImGui::SameLine();
                ImGui::PopStyleColor();
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.8f, 0.1f, 1.0f));
                ImGui::Button("Y"); ImGui::SameLine(); ImGui::DragFloat(" ##posY", &f); ImGui::SameLine();
                ImGui::PopStyleColor();
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.8f, 1.0f));
                ImGui::Button("Z"); ImGui::SameLine(); ImGui::DragFloat(" ##posZ", &f);
                ImGui::PopStyleColor();
                ImGui::Separator();

                ImGui::Text("Rotation");
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
                ImGui::Button("X"); ImGui::SameLine(); ImGui::DragFloat(" ##rotX", &f); ImGui::SameLine();
                ImGui::PopStyleColor();
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.8f, 0.1f, 1.0f));
                ImGui::Button("Y"); ImGui::SameLine(); ImGui::DragFloat(" ##rotY", &f); ImGui::SameLine();
                ImGui::PopStyleColor();
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.8f, 1.0f));
                ImGui::Button("Z"); ImGui::SameLine(); ImGui::DragFloat(" ##rotZ", &f);
                ImGui::PopStyleColor();
                ImGui::Separator();

                ImGui::Text("Scale");
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
                ImGui::Button("X"); ImGui::SameLine(); ImGui::DragFloat(" ##scaX", &f); ImGui::SameLine();
                ImGui::PopStyleColor();
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.8f, 0.1f, 1.0f));
                ImGui::Button("Y"); ImGui::SameLine(); ImGui::DragFloat(" ##scaY", &f); ImGui::SameLine();
                ImGui::PopStyleColor();
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.8f, 1.0f));
                ImGui::Button("Z"); ImGui::SameLine(); ImGui::DragFloat(" ##scaZ", &f);
                ImGui::PopStyleColor();
                ImGui::Separator();
                ImGui::Button("To Center", ImVec2(100, 25));
            }
            if (ImGui::CollapsingHeader("Material")) {
                ImGui::Text("Diffuse:");
                ImGui::ColorEdit3("dif", (float*)&clear_color);
                ImGui::Image(NULL, ImVec2(120, 120), ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0), ImVec4(1, 1, 1, 1));
                ImGui::Separator();
                ImGui::Text("Specular:");
                ImGui::ColorEdit3("spe", (float*)&clear_color2);
                ImGui::Image(NULL, ImVec2(120, 120), ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0), ImVec4(1, 1, 1, 1));
                ImGui::Separator();
                static int currents = 0;
                const char* items[] = { "light_shader","phong_shader" };
                ImGui::Combo("Shader Pipeline", &currents, items, 2);
            }
            if (ImGui::CollapsingHeader("Scripts")) {

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

// Main code
int main(int, char**)
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Create window with graphics context
    glfwWindowHint(GLFW_MAXIMIZED, true);
    GLFWwindow* window = glfwCreateWindow(1920, 960, "Dear ImGui GLFW+OpenGL2 example", NULL, NULL); // glfwGetPrimaryMonitor()
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

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
    glClearColor(0,0,0,0);
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ShowSceneObj();
        ShowResources();
        ShowRenderSettings();
        ShowNodeProperties();
        ShowConsoles();
        //ImGui::ShowStyleEditor();
        if (show_settings_window) {
            ImGui::Begin("Settings", &show_settings_window, 0);
            ImGui::ShowStyleEditor();
            ImGui::End();
        }
        //if (show_demo_window)
        //ImGui::ShowDemoWindow(&show_demo_window);
        ImGui::PushFont(font2);
        ShowExampleAppMainMenuBar();
        ImGui::PopFont();

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);

        // If you are using this code with non-legacy OpenGL header/contexts (which you should not, prefer using imgui_impl_opengl3.cpp!!),
        // you may need to backup/reset/restore other state, e.g. for current shader using the commented lines below.
        //GLint last_program;
        //glGetIntegerv(GL_CURRENT_PROGRAM, &last_program);
        //glUseProgram(0);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        //glUseProgram(last_program);

        glfwMakeContextCurrent(window);
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
