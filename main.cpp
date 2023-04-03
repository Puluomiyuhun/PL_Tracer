#include "SampleRenderer.h"

// our helper library for window handling
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>

namespace osc {

    //继承optix course提供的一个glfwindow库中的类，这个类的窗口已经写好了鼠标键盘交互的逻辑
    struct SampleWindow : public GLFCameraWindow
    {
        SampleWindow(const std::string& title,
            const Model* model,
            const Camera& camera,
            const QuadLight& light,
            const float worldScale)
            : GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale),
            sample(model, light)
        {
            sample.setCamera(camera);
        }

        /*override一下鼠标交互逻辑 */
        virtual void mouseMotion(const vec2i& newPos) override
        {
            vec2i windowSize;
            glfwGetWindowSize(handle, &windowSize.x, &windowSize.y);

            if (isPressed.leftButton && cameraFrameManip)
                cameraFrameManip->mouseDragLeft(vec2f(newPos - lastMousePos) / vec2f(windowSize) * 0.6f);
            if (isPressed.rightButton && cameraFrameManip)
                cameraFrameManip->mouseDragRight(vec2f(newPos - lastMousePos) / vec2f(windowSize) * 0.1f);
            if (isPressed.middleButton && cameraFrameManip)
                cameraFrameManip->mouseDragMiddle(vec2f(newPos - lastMousePos) / vec2f(windowSize) * 0.1f);
            lastMousePos = newPos;
        }

        /*override一下渲染逻辑，设置相机并调用渲染器的render函数*/
        virtual void render() override
        {
            if (cameraFrame.modified) {
                sample.setCamera(Camera{ cameraFrame.get_from(),
                                         cameraFrame.get_at(),
                                         cameraFrame.get_up() });
                cameraFrame.modified = false;
            }
            sample.render();
        }

        /*override一下绘制逻辑，在窗口开一张纹理铺满画面，然后把渲染结果贴上去*/
        virtual void draw() override
        {
            sample.downloadPixels(pixels.data());
            if (fbTexture == 0)
                glGenTextures(1, &fbTexture);

            glBindTexture(GL_TEXTURE_2D, fbTexture);
            GLenum texFormat = GL_RGBA;
            GLenum texelType = GL_FLOAT;
            glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                texelType, pixels.data());

            glDisable(GL_LIGHTING);
            glColor3f(1, 1, 1);

            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, fbTexture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            glDisable(GL_DEPTH_TEST);

            glViewport(0, 0, fbSize.x, fbSize.y);

            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

            glBegin(GL_QUADS);
            {
                glTexCoord2f(0.f, 0.f);
                glVertex3f(0.f, 0.f, 0.f);

                glTexCoord2f(0.f, 1.f);
                glVertex3f(0.f, (float)fbSize.y, 0.f);

                glTexCoord2f(1.f, 1.f);
                glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

                glTexCoord2f(1.f, 0.f);
                glVertex3f((float)fbSize.x, 0.f, 0.f);
            }
            glEnd();
        }

        virtual void resize(const vec2i& newSize)
        {
            fbSize = newSize;
            sample.resize(newSize);
            pixels.resize(newSize.x * newSize.y);
        }

        vec2i                 fbSize;
        GLuint                fbTexture{ 0 };
        SampleRenderer        sample;
        std::vector<vec4f> pixels;
    };


    extern "C" int main(int ac, char** av)
    {
        /*下面加载模型，要根据自己的模型目录来写路径*/
        try {
            //Model* model = loadOBJ("../x64/Debug/models/sponza/sponza.obj", DIFFUSE);
            //Model* model = loadOBJ("../x64/Debug/models/fireplace_room/fireplace_room.obj", DIFFUSE);
            //Model* model = loadOBJ("../x64/Debug/models/bedroom/iscv2.obj", DIFFUSE);
            //Model* model = loadOBJ("../x64/Debug/models/breakfast_room/breakfast_room.obj", METAL);
            //Model* model = loadOBJ("../x64/Debug/models/conference/conference.obj", METAL);
            //Model* model = loadOBJ("../x64/Debug/models/gun/gun.obj", DIFFUSE);
            Model* model = loadOBJ("../x64/Debug/models/dress/dress.obj", METAL);
            //Model* model = loadOBJ("../x64/Debug/models/office/office.obj", METAL);
            //Model* model = loadOBJ("../x64/Debug/models/spaceship/spaceship.obj", METAL);
            //Model* model = loadOBJ("../x64/Debug/models/ball/ball.obj", METAL);
            loadEnvmap(model, "../x64/Debug/env/farm.hdr");
            Camera camera = { vec3f(-1293.07f, 154.681f, -0.7304f),
                model->bounds.center() - vec3f(0,400,0),
                vec3f(0.f,1.f,0.f) };
            const float light_size = 200.f;
            QuadLight light = { vec3f(-1000 - light_size,800,-light_size),
               vec3f(2.f * light_size,0,0),
               vec3f(0,0,2.f * light_size),
               vec3f(3000000.f) };
            const float worldScale = length(model->bounds.span());
            
            /*Model* model = loadOBJ("../x64/Debug/models/bmw/bmw.obj", DIFFUSE);
            loadEnvmap(model, "../x64/Debug/env/church.hdr");
            Camera camera = { vec3f(-1000.0f, 1000.0f, 0),
                model->bounds.center() - vec3f(0,400,0),
                vec3f(0.f,1.f,0.f) };
            const float light_size = 200.f;
            QuadLight light = { vec3f(-1000 - light_size,800,-light_size),
               vec3f(2.f * light_size,0,0),
               vec3f(0,0,2.f * light_size),
               vec3f(3000000.f) };
            const float worldScale = length(model->bounds.span());*/

            SampleWindow* window = new SampleWindow("PL Tracer", model, camera, light, worldScale);
            window->enableFlyMode();
            window->run();

        }
        catch (std::runtime_error& e) {
            std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
                << GDT_TERMINAL_DEFAULT << std::endl;
            std::cout << "Did you forget to copy sponza.obj and sponza.mtl into your optix7course/models directory?" << std::endl;
            exit(1);

        }
        return 0;
    }

}
