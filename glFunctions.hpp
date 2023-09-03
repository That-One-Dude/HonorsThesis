#include <SFML/OpenGL.hpp>
#include <GL/glext.h>

#ifndef GL_FUNCTIONS_HPP
#define GL_FUNCTIONS_HPP

#ifdef FROM_GL_FUNCTIONS_CPP
    #define OPENGL_FUNCTION(type, name) type name;
#else
    #define OPENGL_FUNCTION(type, name) extern type name;
#endif


void initGLFunctions();

// Buffers
#include "glFunctionList.cpp"

// Vertex Arrays


#endif