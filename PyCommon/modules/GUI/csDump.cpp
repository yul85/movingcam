#include "stdafx.h"

#include "csDump.h"
#include "lodepng.h"

#include <fstream>
#include <sstream>
#include <iostream>

#ifdef WIN32
#include <windows.h>
#endif
#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif


BOOST_PYTHON_MODULE(csDump)
{
   def("dump_png", dump_png);
}


void dump_png(char* filename, int w, int h)
{
    int image_size = 3*w*h;
    unsigned char *image_tmp = new unsigned char[image_size];
    unsigned char *image = new unsigned char[image_size];

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK_LEFT);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, image_tmp);

    for(int i=0; i<h; i++)
        memcpy(&image[3*w*(h-1-i)], &image_tmp[3*w*i], 3*w*sizeof(unsigned char));

    unsigned int error = lodepng_encode24_file(filename, image, w, h);

    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    if (image_tmp) delete [] image_tmp;
    if (image) delete [] image;
}
