#include <iostream>
#include <cmath>
#include <SFML/Graphics.hpp>
#include <chrono>
#include "glFunctions.hpp"
//#include "nonEuclidianSpace.hpp"
#include "glMath.hpp"

/*struct vec4 {
    float x, y, z, w;

    vec4();
    vec4(float x, float y, float z, float w): x(x), y(y), z(z), w(w) {}
};

mat3 compose(mat3 lhs, mat3 rhs) {
    return mat3(lhs.m11 * rhs.m11 + lhs.m12 * rhs.m21 + lhs.m13 * rhs.m31, lhs.m11 * rhs.m12 + lhs.m12 * rhs.m22 + lhs.m13 * rhs.m32, lhs.m11 * rhs.m13 + lhs.m12 * rhs.m23 + lhs.m13 * rhs.m33,
        lhs.m21 * rhs.m11 + lhs.m22 * rhs.m21 + lhs.m23 * rhs.m31, lhs.m21 * rhs.m12 + lhs.m22 * rhs.m22 + lhs.m23 * rhs.m32, lhs.m21 * rhs.m13 + lhs.m22 * rhs.m23 + lhs.m23 * rhs.m33,
        lhs.m31 * rhs.m11 + lhs.m32 * rhs.m21 + lhs.m33 * rhs.m31, lhs.m31 * rhs.m12 + lhs.m32 * rhs.m22 + lhs.m33 * rhs.m32, lhs.m31 * rhs.m13 + lhs.m32 * rhs.m23 + lhs.m33 * rhs.m33);
}

mat3 add(mat3 lhs, mat3 rhs) {
    return mat3(lhs.m11 + rhs.m11, lhs.m12 + rhs.m12, lhs.m13 + rhs.m13,
        lhs.m21 + rhs.m21, lhs.m22 + rhs.m22, lhs.m23 + rhs.m23,
        lhs.m31 + rhs.m31, lhs.m32 + rhs.m32, lhs.m33 + rhs.m33);
}

vec3 apply(mat3 transform, vec3 vector) {
    return vec3(transform.m11 * vector.x + transform.m12 * vector.y + transform.m13 * vector.z,
        transform.m21 * vector.x + transform.m22 * vector.y + transform.m23 * vector.z,
        transform.m31 * vector.x + transform.m32 * vector.y + transform.m33 * vector.z);
}

mat3 rotation(float radians) {
    return mat3(cosf(radians), -sinf(radians), 0,
        sinf(radians), cosf(radians), 0,
        0, 0, 1);
}

mat3 hyperTranslation(vec2 offset) {
    float magnitude = sqrtf(offset.x * offset.x + offset.y * offset.y);
    if (magnitude > 0) {
        float Cos = offset.x / magnitude, Sin = offset.y / magnitude;
        float CosH = coshf(magnitude) - 1, SinH = sinhf(magnitude);
        return mat3(Cos * Cos * CosH + 1, Cos * Sin * CosH, Cos * SinH,
            Cos * Sin * CosH, Sin * Sin * CosH + 1, Sin * SinH,
            Cos * SinH, Sin * SinH, CosH + 1);
    } else {
        return mat3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    }
}

mat3 sphericalTranslate(vec2 offset) {
    float magnitude = sqrtf(offset.x * offset.x + offset.y * offset.y);
    if (magnitude > 0) {
        float Cos = offset.x / magnitude, Sin = offset.y / magnitude;
        float CosH = cosf(magnitude) - 1, SinH = sinf(magnitude);
        return mat3(Cos * Cos * CosH + 1, Cos * Sin * CosH, Cos * SinH,
            Cos * Sin * CosH, Sin * Sin * CosH + 1, Sin * SinH,
            -Cos * SinH, -Sin * SinH, CosH + 1);
    } else {
        return mat3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    }
}*/

struct Vertex {
    vec3 position;
    vec3 normal;
    vec2 texCoords;
    vec3 tangent;
    vec3 bitangent;

    Vertex() {}
    Vertex(vec3 position, vec3 normal, vec2 texCoords, vec3 tangent, vec3 bitangent): position(position), normal(normal), texCoords(texCoords), tangent(tangent), bitangent(bitangent) {}
};

int main() {

    sf::Window window(sf::VideoMode(800, 800), "Hyperbolic Test");

    initGLFunctions();

    /*const GLchar* vertexShaderSource = "#version 330 core\n"
        "#extension GL_ARB_gpu_shader_fp64 : enable\n"
        "uniform mat3 offset;\n"
        "uniform mat2 transform;\n"
        "layout (location = 0) in vec3 aPos;\n"

        // Hyperbolic:
        "vec2 toEuclideanCoordinates(vec3 from) {\n"
        "   if (from.z < 1) from.z = 1;\n"
        "   float magnitude = acosh(from.z);\n"
        "   float s = sinh(magnitude);\n"
        "   if (s < 0.01f) s = 0.01f;\n"
        "   magnitude = magnitude / s;\n"
        "   return vec2(from.x * magnitude, from.y * magnitude);\n"
        "}\n"

        // Spherical:
        /*"vec2 toEuclideanCoordinates(vec3 from) {\n"
        "   float magnitude = acos(from.z);\n"
        "   magnitude = magnitude / sin(magnitude);\n"
        "   return vec2(from.x * magnitude, from.y * magnitude);\n"
        "}\n"*//*

        "void main()\n"
        "{\n"
        "   vec3 pos = offset * aPos;\n"
        "   vec2 truePos = transform * toEuclideanCoordinates(pos);\n"
        "   gl_Position = vec4(truePos.x, truePos.y, 0.0, 1.0);\n"
        "}\0";*/
    
    const GLchar* vertexShaderSource = "#version 440 core\n"
        "layout (location = 0) in vec3 aPos;\n"
        "layout (location = 1) in vec3 aNormal;\n"
        "layout (location = 2) in vec2 aTexCoords;\n"
        "layout (location = 3) in vec3 aTangent;\n"
        "layout (location = 4) in vec3 aBitangent;\n"

        "out VS_OUT {\n"
        "   vec3 FragPos;\n"
        "   vec2 TexCoords;\n"
        "   vec3 TangentLightPos;\n"
        "   vec3 TangentViewPos;\n"
        "   vec3 TangentFragPos;\n"
        "} vs_out;\n"

        "uniform mat4 camera;\n"
        "uniform vec3 viewPos;\n"

        "void main()\n"
        "{\n"
        "   vs_out.FragPos = aPos;\n"
        "   vs_out.TexCoords = aTexCoords;\n"
            
        "   vec3 T = normalize(aTangent);\n"
        "   vec3 B = normalize(aBitangent);\n"
        "   vec3 N = normalize(aNormal);\n"
        "   mat3 TBN = transpose(mat3(T, B, N));\n"

        "   vs_out.TangentViewPos  = TBN * viewPos;\n"
        "   vs_out.TangentFragPos  = TBN * vs_out.FragPos;\n"
            
        "   gl_Position = camera * vec4(aPos, 1.0);\n"
        //"gl_Position = vec4(aPos.x * 2, aPos.y * 2, 1.0f, 1.0f);\n"
        "}\0";

    const GLchar* fragmentShaderSource = "#version 440 core\n"
        "out vec4 FragColor;\n"

        "uniform sampler2D depthMap;\n"
        "uniform sampler2D diffuseMap;\n"

        "layout (std140, binding = 0) uniform Coefficients {\n"
        "   float coefficients[64];\n"
        "};\n"

        "uniform float heightScale;\n"

        "uniform vec3 viewDirection;\n"

        "in VS_OUT {\n"
        "   vec3 FragPos;\n"
        "   vec2 TexCoords;\n"
        "   vec3 TangentLightPos;\n"
        "   vec3 TangentViewPos;\n"
        "   vec3 TangentFragPos;\n"
        "} fs_in;\n"

        "void main() {\n"
        //"   vec3 viewDir = viewDirection;//normalize(fs_in.TangentViewPos - fs_in.TangentFragPos);\n"
    
        //"   vec2 texCoords = ParallaxMapping(fs_in.TexCoords,  viewDir);\n"
        //"   texCoords = ((texCoords - fs_in.TexCoords));\n"
        //"   float magnitude = sqrt(texCoords.x * texCoords.x + texCoords.y * texCoords.y);\n"
        //"   FragColor = vec4(color.r, color.g, color.b, 1.0f);\n"
        //"   FragColor = vec4((magnitude * abs(dot(vec3(0.0, 0.0, 1.0), viewDir))) / heightScale, (magnitude * abs(dot(vec3(0.0, 0.0, 1.0), viewDir))) / heightScale, 0.5f, 1.0f);\n"
        //"   FragColor = vec4(vec2(0.5f, 0.5f) + (texCoords * abs(dot(vec3(0.0, 0.0, 1.0), viewDir)) / heightScale) / 2, 0.5f, 1.0f);\n"
        //"   FragColor = texture(depthMap, fs_in.TexCoords);\n"
        //"   FragColor = texture(diffuseMap, texCoords);\n"
        "   float value = 0.0f;\n"
        "   for (int i = 0; i < 8; i++) {\n"
        "       float inputX = 1.0f;"
        "       if (i > 0) {\n"
        "           float uPos = fs_in.TexCoords.x * i + 1.0f;\n"
        "           inputX = mod(uPos, 2.0f) - 1.0f;\n"
        "           inputX = (1 - 2 * abs(inputX));\n"
        "       }\n"
        "       for (int j = 0; j < 8; j++) {\n"
        "           float inputY = 1.0f;"
        "           if (j > 0) {\n"
        "               float vPos = fs_in.TexCoords.y * j + 1.0f;\n"
        "               inputY = mod(vPos, 2.0f) - 1.0f;\n"
        "               inputY = (1 - 2 * abs(inputY));\n"
        "           }\n"
        "           value += coefficients[i + 8 * j] * inputX * inputY;\n"
        "       }\n"
        "   }\n"
        "   FragColor = vec4(value, value, value, 0.0f);\n"
        "}\0";

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    glUseProgram(shaderProgram);

    int cameraUniformLocation = glGetUniformLocation(shaderProgram, "camera");
    int viewPosUniformLocation = glGetUniformLocation(shaderProgram, "viewPos");
    int viewDirUniformLocation = glGetUniformLocation(shaderProgram, "viewDirection");
    int heightUniformLocation = glGetUniformLocation(shaderProgram, "heightScale");
    int depthMapUniformLocation = glGetUniformLocation(shaderProgram, "depthMap");
    int diffuseMapUniformLocation = glGetUniformLocation(shaderProgram, "diffuseMap");
    //int colorUniformLocation = glGetUniformLocation(shaderProgram, "color");

    glUniform1f(heightUniformLocation, 0.5f);
    glUniform1i(depthMapUniformLocation, 0);
    glUniform1i(diffuseMapUniformLocation, 1);

    sf::Texture texture;
    texture.loadFromFile("textures/parallaxMap.jpg");
    sf::Texture diffuse;
    diffuse.loadFromFile("textures/diffuseMap.jpg");

    glActiveTexture(GL_TEXTURE0);
    sf::Texture::bind(&texture);

    glActiveTexture(GL_TEXTURE1);
    sf::Texture::bind(&diffuse);

    sf::Image parallaxImage = texture.copyToImage();

    GLuint uniformBuffer;
    glGenBuffers(1, &uniformBuffer);
    float coefficients[256];
    const float multiplier = sqrtf(3 / 2);
    const float constCoefficient = sqrtf(2) / 2;
    const int numPixels = parallaxImage.getSize().x * parallaxImage.getSize().y;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            float integral = 0.0f;
            for (int xPixel = 0; xPixel < texture.getSize().x; xPixel++) {
                float funcX = constCoefficient;
                if (i > 0) {
                    funcX = (xPixel + 0.5f) / texture.getSize().x;
                    funcX *= i;
                    funcX = fmodf(funcX + 1.0f, 2.0f) - 1.0f;
                    funcX = 1.0f - 2.0f * fabsf(funcX);
                    funcX *= multiplier;
                }
                for (int yPixel = 0; yPixel < texture.getSize().y; yPixel++) {
                    float funcY = constCoefficient;
                    if (j > 0) {
                        funcY = (yPixel + 0.5f) / texture.getSize().y;
                        funcY *= j;
                        funcY = fmodf(funcY + 1.0f, 2.0f) - 1.0f;
                        funcY = 1.0f - 2.0f * fabsf(funcY);
                        funcY *= multiplier;
                    }
                    sf::Color pixel = parallaxImage.getPixel(xPixel, yPixel);
                    integral += (pixel.r / 256.0f) * funcX * funcY;
                }
            }
            coefficients[4 * (i + 8 * j)] = integral / numPixels;
            //std::cout << integral << std::endl;
            //coefficients[4 * (i + 8 * j)] = 0.0f;
        }
    }

    //coefficients[0] = 0.5f;
    //coefficients[72] = 0.5f;

    glBindBuffer(GL_UNIFORM_BUFFER, uniformBuffer);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(coefficients), coefficients, GL_STATIC_DRAW);

    glBindBufferBase(GL_UNIFORM_BUFFER, 0, uniformBuffer);

    GLuint vbo;
    glGenBuffers(1, &vbo);
    GLuint vao;
    glGenVertexArrays(1, &vao);

    float width = M_PI_2;

    /*vec2 top = HyperbolicCompose(vec2(-width, 0), vec2(0, width));
    vec2 bottom = HyperbolicCompose(vec2(width, 0), vec2(0, -width));

    vec2 square[] {
        vec2(-bottom.x, bottom.y),
        vec2(-width, 0),    
        bottom,

        vec2(-width, 0),    
        bottom,
        vec2(width, 0),  

        vec2(-width, 0),
        top,
        vec2(width, 0),

        top,
        vec2(width, 0),
        vec2(-top.x, top.y),
    };*/

    Vertex topLeft = Vertex(vec3(0.5f, 0.5f, 0.5f), vec3(0.0f, 0.0f, 1.0f), vec2(1.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f));
    Vertex topRight = Vertex(vec3(-0.5f, 0.5f, 0.5f), vec3(0.0f, 0.0f, 1.0f), vec2(0.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f));
    Vertex bottomLeft = Vertex(vec3(0.5f, -0.5f, 0.5f), vec3(0.0f, 0.0f, 1.0f), vec2(1.0f, 1.0f), vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f));
    Vertex bottomRight = Vertex(vec3(-0.5f, -0.5f, 0.5f), vec3(0.0f, 0.0f, 1.0f), vec2(0.0f, 1.0f), vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f));

    Vertex square[] {
        topLeft,
        topRight,
        bottomLeft,

        topRight,
        bottomLeft,
        bottomRight
    };

    //for (int i = 0; i < sizeof(square) / sizeof(vec2); i++) {
        //square[i] = EuclidianTransformToHyperbolicTransform(square[i]);
    //}

    /*struct Pos {
        mat4 offset;
        vec3 color;

        Pos(mat4 offset, vec3 color): offset(offset), color(color) {}
    } poses[]{
        Pos(mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1), vec3(1.0f, 0.0f, 0.0f)),
        Pos(sphericalTranslation(vec3(width, 0, 0)), vec3(0.0f, 1.0f, 0.0f)),
        //Pos(sphericalTranslation(vec3(width, 0, 0)) * sphericalTranslation(vec3(0, 0, width)), vec3(0.0f, 0.0f, 1.0f)),
        Pos(sphericalTranslation(vec3(0, 0, width)), vec3(0.0f, 0.0f, 1.0f)),
        //Pos(sphericalTranslation(vec3(0, 0, width)) * sphericalTranslation(vec3(width, 0, 0)), vec3(1.0f, 0.0f, 0.0f))
        Pos(sphericalTranslation(vec3(-width, 0, 0)), vec3(1.0f, 0.5f, 0.0f)),
        Pos(sphericalTranslation(vec3(0, 0, -width)), vec3(0.0f, 1.0f, 0.5f)),
        Pos(sphericalTranslation(vec3(0, 0, 2 * width)), vec3(0.5f, 0.0f, 1.0f)),
    };

    Vertex vertices[sizeof(poses) * sizeof(square) / (sizeof(Pos) * sizeof(Vertex))];

    for (int i = 0; i < sizeof(poses) / sizeof(Pos); i++) {
        for (int j = 0; j < sizeof(square) / sizeof(Vertex); j++) {
            vertices[(i * sizeof(square) / sizeof(Vertex)) + j].position = poses[i].offset * square[j].position;
            vertices[(i * sizeof(square) / sizeof(Vertex)) + j].color = poses[i].color;
        }
    }

    for (int i = 0; i < sizeof(vertices) / sizeof(Vertex); i++) {
        //std::cout << vertices[i].x << ", " << vertices[i].y << ", " << vertices[i].z << std::endl;
    }*/

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(square), square, GL_STATIC_DRAW);

    size_t attributeOffset = 0;
    glVertexAttribPointer(0, 3, GL_FLOAT, false, sizeof(Vertex), (void*)attributeOffset);
    glEnableVertexAttribArray(0);
    attributeOffset += sizeof(vec3);
    glVertexAttribPointer(1, 3, GL_FLOAT, false, sizeof(Vertex), (void*)attributeOffset);
    glEnableVertexAttribArray(1);
    attributeOffset += sizeof(vec3);
    glVertexAttribPointer(2, 2, GL_FLOAT, false, sizeof(Vertex), (void*)attributeOffset);
    glEnableVertexAttribArray(2);
    attributeOffset += sizeof(vec2);
    glVertexAttribPointer(3, 3, GL_FLOAT, false, sizeof(Vertex), (void*)attributeOffset);
    glEnableVertexAttribArray(3);
    attributeOffset += sizeof(vec3);
    glVertexAttribPointer(4, 3, GL_FLOAT, false, sizeof(Vertex), (void*)attributeOffset);
    glEnableVertexAttribArray(4);
    attributeOffset += sizeof(vec3);

    glUseProgram(shaderProgram);

    glCullFace(GL_NONE);
    
    mat4 cam = perspective(M_PI_4, 1.0f, 0.1f, 10.0f);//mat2(0.25f,0.0f,0.0f,0.25f);
    mat4 offset = euclideanTranslation(vec3(0, 0, 0));//mat3(1, 0, 0, 0, 1, 0, 0, 0, 1);

    glUniformMatrix4fv(cameraUniformLocation, 1, false, (float*)&cam);
    //glUniform3f(colorUniformLocation, 1.0f, 0.5f, 0.5f);

    float sensitivity = 0.01f;
    vec2 cameraRotation = vec2(0, 0);
    vec4 motion = vec4(0, 0, 0, 1);
    float speed = 1.0f;
    bool motionRight = false, motionLeft = false, motionUp = false, motionDown = false;
    bool motionDirectlyUp = false, motionDirectlyDown = false;

    std::chrono::steady_clock::time_point start, end;

    end = std::chrono::steady_clock::now();

    bool isPaused = false;

    while (window.isOpen()) {

        start = end;
        end = std::chrono::steady_clock::now();
        float deltaT = (static_cast<std::chrono::duration<float>>(end - start)).count();

        if (!window.hasFocus()) {
            isPaused = true;
        }
        if (!isPaused) {
            sf::Mouse::setPosition(sf::Vector2i(window.getSize().x / 2, window.getSize().y / 2), window);
        }


        motion = vec4(0, 0, 0, 1);

        if (motionRight) {
            motion.x -= speed * deltaT;
        }
        if (motionLeft) {
            motion.x += speed * deltaT;
        }
        if (motionUp) {
            motion.z += speed * deltaT;
        }
        if (motionDown) {
            motion.z -= speed * deltaT;
        }
        if (motionDirectlyUp) {
            motion.y -= speed * deltaT;
        }
        if (motionDirectlyDown) {
            motion.y += speed * deltaT;
        }

        mat4 camRotation = rotationY(-cameraRotation.x);
        motion = camRotation * motion;


        camRotation = rotationY(cameraRotation.x);

        mat4 translation = euclideanTranslation(vec3(motion.x, motion.y, motion.z));//motion = EuclidianTransformToHyperbolicTransform(motion);

        offset = translation * offset;

        camRotation = rotationX(-cameraRotation.y) * camRotation;

        vec4 trueCamDir = camRotation * vec4(0, 0, 1, 0);

        mat4 camPos = euclideanTranslation(vec3(0, -0.5, 0)) * offset;
        mat4 camTransform = cam * camRotation * camPos;

        vec4 trueCamPos = camPos * vec4(0, 0, 0, -1);

        //glUniformMatrix4fv(offsetUniformLocation, 1, false, (float*)&camPos);
        glUniformMatrix4fv(cameraUniformLocation, 1, false, (float*)&camTransform);
        glUniform3f(viewPosUniformLocation, trueCamPos.x, trueCamPos.y, trueCamPos.z);
        glUniform3f(viewDirUniformLocation, trueCamDir.x, trueCamDir.y, trueCamDir.z);

        glClear(GL_COLOR_BUFFER_BIT);

        glDrawArrays(GL_TRIANGLES, 0, sizeof(square) / sizeof(Vertex));

        window.display();

        sf::Event event;
        while (window.pollEvent(event)) {

            if (event.type == sf::Event::EventType::Closed) {
                window.close();
            }

            if (event.type == sf::Event::EventType::KeyPressed) {
                switch (event.key.code) {
                    case sf::Keyboard::W: {
                        motionUp = true;
                        break;
                    }
                    case sf::Keyboard::S: {
                        motionDown = true;
                        break;
                    }
                    case sf::Keyboard::D: {
                        motionRight = true;
                        break;
                    }
                    case sf::Keyboard::A: {
                        motionLeft = true;
                        break;
                    }
                    case sf::Keyboard::Space: {
                        motionDirectlyUp = true;
                        break;
                    }
                    case sf::Keyboard::LShift: {
                        motionDirectlyDown = true;
                        break;
                    }
                    case sf::Keyboard::Escape: {
                        isPaused = !isPaused;
                        break;
                    }
                }
            }

            if (event.type == sf::Event::EventType::KeyReleased) {
                switch (event.key.code) {
                    case sf::Keyboard::W: {
                        motionUp = false;
                        break;
                    }
                    case sf::Keyboard::S: {
                        motionDown = false;
                        break;
                    }
                    case sf::Keyboard::D: {
                        motionRight = false;
                        break;
                    }
                    case sf::Keyboard::A: {
                        motionLeft = false;
                        break;
                    }
                    case sf::Keyboard::Space: {
                        motionDirectlyUp = false;
                        break;
                    }
                    case sf::Keyboard::LShift: {
                        motionDirectlyDown = false;
                        break;
                    }
                }
            }

            if (event.type == sf::Event::MouseMoved) {
                int oldX = window.getSize().x / 2, oldY = window.getSize().y / 2;
                cameraRotation.x += (event.mouseMove.x - oldX) * sensitivity;
                cameraRotation.y += (event.mouseMove.y - oldY) * sensitivity;
                while (cameraRotation.x > M_PI) {
                    cameraRotation.x -= 2 * M_PI;
                }
                while (cameraRotation.x <= M_PI) {
                    cameraRotation.x += 2 * M_PI;
                }
                if (cameraRotation.y > M_PI_2 - 0.1f) {
                    cameraRotation.y = M_PI_2 - 0.1f;
                }
                if (cameraRotation.y < -M_PI_2 + 0.1f) {
                    cameraRotation.y = -M_PI_2 + 0.1f;
                }
            }

        }
    }

}