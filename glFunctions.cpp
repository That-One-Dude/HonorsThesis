#define FROM_GL_FUNCTIONS_CPP
#include "glFunctions.hpp"

void* GetAnyGLFuncAddress(const char* name)
{
  void* p = (void*)wglGetProcAddress(name);
  if(p == 0 ||
    (p == (void*)0x1) || (p == (void*)0x2) || (p == (void*)0x3) ||
    (p == (void*)-1) )
  {
    HMODULE module = LoadLibraryA("opengl32.dll");
    p = (void*)GetProcAddress(module, name);
  }

  return p;
}

#define OPENGL_FUNCTION(type, name) name = (type)GetAnyGLFuncAddress(#name);

void initGLFunctions() {
    #include "glFunctionList.cpp"
}
